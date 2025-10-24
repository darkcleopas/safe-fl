#!/usr/bin/env python3
"""
Script completo para comparação de defesas em Federated Learning.
Este script foi refatorado para separar a execução da análise.
- ExperimentRunner: Lida com a geração de configurações e execução de simulações.
- ResultAnalyzer: Lida com o carregamento, análise e visualização de resultados.
"""

import os
import sys
import yaml
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
import re
import ast


# Adicionar diretório raiz ao path, se necessário
# (Assumindo que fl_simulator está no mesmo diretório ou no PYTHONPATH)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importe o simulador (certifique-se de que o import funcione no seu ambiente)
try:
    from fl_simulator import FLSimulator
except ImportError:
    print("AVISO: Não foi possível importar FLSimulator. O modo de execução não funcionará.")
    FLSimulator = None


def slugify(text):
    """Converte um texto para um formato 'slug' seguro para nomes de arquivos."""
    text = text.lower()
    text = re.sub(r'\s+', '_', text)  # Substitui espaços por underscores
    text = re.sub(r'[^\w\-]+', '', text) # Remove caracteres não alfanuméricos
    return text


class ResultAnalyzer:
    """
    Classe para carregar, analisar e visualizar resultados de experimentos.
    """
    def __init__(self, simulation_dir):
        """
        Inicializa o analisador com o diretório base da simulação.

        Args:
            simulation_dir (str or Path): O diretório da simulação contendo a pasta 'results'.
        """
        self.base_dir = Path(simulation_dir)
        self.results_dir = self.base_dir / 'results'
        self.plots_dir = self.base_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Diretório de resultados não encontrado: {self.results_dir}")

        self.all_results = {}
        self.metadata = {}
        print(f"📊 Analisador iniciado para o diretório: {self.base_dir}")

    def load_results(self):
        """
        Carrega os resultados a partir de um arquivo de resumo JSON ou varrendo os diretórios.
        """
        summary_file = self.results_dir / 'complete_results.json'
        if summary_file.exists():
            print(f"📄 Carregando resultados do arquivo de resumo: {summary_file}")
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            self.all_results = summary_data.get('results', {})
            self.metadata = summary_data.get('experiment_info', {})
            print(f"✅ {len(self.all_results)} resultados carregados.")
        else:
            print("⚠️ Arquivo de resumo não encontrado. Varrendo diretórios de experimentos...")
            for exp_dir in self.results_dir.iterdir():
                if exp_dir.is_dir():
                    metrics_file = exp_dir / 'metrics.json'
                    config_file = exp_dir / 'config.yaml'

                    if metrics_file.exists() and config_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                            with open(config_file, 'r') as f:
                                config = yaml.safe_load(f)
                            
                            malicious_clients = []
                            malicious_file = exp_dir / 'malicious_clients.json'
                            if malicious_file.exists():
                                with open(malicious_file, 'r') as f:
                                    malicious_data = json.load(f)
                                    malicious_clients = malicious_data.get('malicious_clients', [])

                            # Extrai metadados do config.yaml
                            exp_name = config.get('experiment', {}).get('name', exp_dir.name)
                            defense_name = "Sem Defesa"
                            if config.get('server', {}).get('defense_pipeline'):
                                defense_name = config.get('experiment', {}).get('defense_name', exp_name.split('_')[2]) # Fallback

                            attack_rate = config.get('clients', {}).get('malicious_percentage', 0.0)
                            sel_frac = config.get('server', {}).get('selection_fraction', 0.0)

                            self.all_results[exp_name] = {
                                'config_path': str(config_file),
                                'defense': defense_name,
                                'attack_rate': attack_rate,
                                'selection_fraction': sel_frac,
                                'malicious_clients': malicious_clients,
                                'metrics': metrics,
                                'final_accuracy': metrics.get('accuracy', [0.0])[-1],
                                'final_loss': metrics.get('loss', [float('inf')])[-1],
                                'final_precision': metrics.get('precision', [0.0])[-1],
                                'final_recall': metrics.get('recall', [0.0])[-1],
                                'final_f1': metrics.get('f1', [0.0])[-1],
                                'success': True,
                            }
                        except Exception as e:
                            print(f"❌ Erro ao processar o diretório {exp_dir.name}: {e}")
            print(f"✅ Varredura concluída. Encontrados {len(self.all_results)} resultados.")
        
        if not self.all_results:
             print("❌ Nenhum resultado para analisar.")
             return False
        return True
    
    def run_full_analysis(self):
        """Executa o pipeline completo de análise: carregar dados, processar e criar plots."""
        if self.load_results():
            # NOVO: Instancia o processador para gerar DataFrame e gráficos
            processor = ResultProcessor(self.all_results, self.plots_dir, self.results_dir)
            processor.generate_and_save_dataframe()
            processor.generate_final_metrics_summary()
            processor.create_all_plots()
        else:
            print("Análise interrompida.")


class ResultProcessor:
    def __init__(self, all_results, plots_dir, results_dir):
        self.all_results = all_results
        self.plots_dir = plots_dir
        self.results_dir = results_dir

    def generate_and_save_dataframe(self):
        """
        Converte os resultados carregados em um DataFrame Pandas e o salva em um arquivo CSV.
        """
        if not self.all_results:
            print("⚠️ Sem dados para criar o DataFrame.")
            return

        all_experiments_data = []
        for exp_id, exp_details in self.all_results.items():
            if not exp_details.get('success', False):
                continue

            metrics = exp_details.get("metrics", {})
            num_rounds = len(metrics.get("rounds", []))

            if num_rounds == 0:
                continue

            # Converte a lista de clientes maliciosos para uma string para fácil armazenamento no CSV
            malicious_clients_str = str(exp_details.get("malicious_clients", []))

            for i in range(num_rounds):
                # Usar .get() com um valor padrão seguro para evitar erros
                default_list = [None] * num_rounds
                default_bool_list = [False] * num_rounds

                round_data = {
                    "experiment_id": exp_id,
                    "defense": exp_details.get("defense"),
                    "attack_rate": exp_details.get("attack_rate"),
                    "selection_fraction": exp_details.get("selection_fraction"),
                    "malicious_clients": malicious_clients_str,
                    "round": metrics.get("rounds")[i],
                    "accuracy": metrics.get("accuracy")[i],
                    "loss": metrics.get("loss")[i],
                    "precision": metrics.get("precision", default_list)[i],
                    "recall": metrics.get("recall", default_list)[i],
                    "f1_score": metrics.get("f1", default_list)[i],
                    "aggregation_time": metrics.get("aggregation_time", default_list)[i],
                    "round_time": metrics.get("round_time", default_list)[i],
                    "communication_bytes": metrics.get("communication_bytes", default_list)[i],
                    "selected_clients": str(metrics.get("selected_clients", default_list)[i]),
                    "aggregated_clients": str(metrics.get("aggregated_clients", default_list)[i]),
                    "model_updated": metrics.get("model_updated", default_bool_list)[i],
                    "final_accuracy": exp_details.get("final_accuracy"),
                    "final_loss": exp_details.get("final_loss"),
                    "final_precision": exp_details.get("final_precision"),
                    "final_recall": exp_details.get("final_recall"),
                    "final_f1_score": exp_details.get("final_f1"),
                    "success": exp_details.get("success"),
                }
                all_experiments_data.append(round_data)

        df = pd.DataFrame(all_experiments_data)
        df_path = self.results_dir / 'results_summary.csv'
        df.to_csv(df_path, index=False)
        print(f"\n✅ DataFrame com {len(df)} linhas salvo em: {df_path}")

    def _calculate_filter_metrics(self, df_experiment):
        """
        Calcula as métricas de classificação para a tarefa de filtragem de clientes.
        - Verdadeiro Positivo (TP): Cliente malicioso foi REJEITADO.
        - Verdadeiro Negativo (TN): Cliente honesto foi ACEITO.
        - Falso Positivo (FP): Cliente honesto foi REJEITADO.
        - Falso Negativo (FN): Cliente malicioso foi ACEITO.
        """
        # Converte a string de clientes maliciosos (que é a mesma para todo o experimento) em um conjunto para busca rápida.
        try:
            malicious_clients_set = set(ast.literal_eval(df_experiment['malicious_clients'].iloc[0]))
        except (ValueError, SyntaxError):
            malicious_clients_set = set() # Lida com casos onde não há clientes maliciosos

        tp, tn, fp, fn = 0, 0, 0, 0

        for _, row in df_experiment.iterrows():
            selected_clients = set(ast.literal_eval(row['selected_clients']))
            aggregated_clients = set(ast.literal_eval(row['aggregated_clients']))

            # Itera sobre todos os clientes que participaram da seleção no round
            for client_id in selected_clients:
                is_malicious = client_id in malicious_clients_set
                was_aggregated = client_id in aggregated_clients

                if is_malicious and not was_aggregated:
                    tp += 1
                elif not is_malicious and was_aggregated:
                    tn += 1
                elif not is_malicious and not was_aggregated:
                    fp += 1
                elif is_malicious and was_aggregated:
                    fn += 1
        
        # Cálculo das métricas de classificação
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'filter_accuracy': accuracy,
            'filter_precision': precision,
            'filter_recall': recall,
            'filter_f1_score': f1_score,
        }

    def _calculate_update_decision_metrics(self, df_experiment):
        """
        Avalia a decisão do servidor de atualizar ou não o modelo.
        - Cenário Ideal: Atualizar (True) se NENHUM malicioso foi agregado; Não atualizar (False) se ALGUM malicioso foi agregado.
        - TP: Decisão correta de NÃO atualizar (havia maliciosos agregados).
        - TN: Decisão correta de ATUALIZAR (não havia maliciosos agregados).
        - FP: Decisão errada de NÃO atualizar (não havia maliciosos agregados).
        - FN: Decisão errada de ATUALIZAR (havia maliciosos agregados).
        """
        try:
            malicious_clients_set = set(ast.literal_eval(df_experiment['malicious_clients'].iloc[0]))
        except (ValueError, SyntaxError):
            malicious_clients_set = set()

        tp, tn, fp, fn = 0, 0, 0, 0
        
        for _, row in df_experiment.iterrows():
            try:
                aggregated_clients = set(ast.literal_eval(row['aggregated_clients']))
            except (ValueError, SyntaxError):
                continue

            malicious_in_aggregation = not malicious_clients_set.isdisjoint(aggregated_clients)
            model_was_updated = row['model_updated']

            if malicious_in_aggregation and not model_was_updated:
                tp += 1
            elif not malicious_in_aggregation and model_was_updated:
                tn += 1
            elif not malicious_in_aggregation and not model_was_updated:
                fp += 1
            elif malicious_in_aggregation and model_was_updated:
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'update_decision_accuracy': accuracy,
            'update_decision_true_positives': tp,
            'update_decision_true_negatives': tn,
            'update_decision_false_positives': fp,
            'update_decision_false_negatives': fn,
        }

    def generate_final_metrics_summary(self):
        """
        Lê o CSV de resumo detalhado, agrega os dados por experimento e salva
        um novo CSV com as métricas finais, ideal para tabelas de dissertação.
        """
        detailed_csv_path = self.results_dir / 'results_summary.csv'
        if not detailed_csv_path.exists():
            print(f"⚠️  Arquivo {detailed_csv_path} não encontrado. Pulando a geração do sumário final.")
            return

        print("\n📄 Gerando CSV com métricas finais agregadas...")
        df = pd.read_csv(detailed_csv_path)
        
        # Agrupa o dataframe por cada experimento único
        grouped_experiments = df.groupby('experiment_id')
        
        final_results = []

        for exp_id, df_exp in grouped_experiments:
            final_row = df_exp.iloc[-1] # Pega a última linha para métricas finais do modelo

            # 1. Métricas do Filtro de Clientes
            filter_metrics = self._calculate_filter_metrics(df_exp)

            # 2. Métricas da Decisão de Atualização do Modelo
            update_decision_metrics = self._calculate_update_decision_metrics(df_exp)
            
            # 3. Métricas de Custo e Tempo
            # Usar .dropna() para evitar erros se houver NaNs
            median_agg_time = df_exp['aggregation_time'].dropna().median()
            std_agg_time = df_exp['aggregation_time'].dropna().std()
            total_time = df_exp['round_time'].sum()
            
            # Monta o dicionário com todos os resultados finais para este experimento
            exp_summary = {
                'experiment_id': exp_id,
                'defense': final_row['defense'],
                'attack_rate': final_row['attack_rate'],
                'selection_fraction': final_row['selection_fraction'],

                # Métricas finais do modelo global
                'final_global_accuracy': final_row['final_accuracy'],
                'final_global_loss': final_row['final_loss'],
                'final_global_precision': final_row['final_precision'],
                'final_global_recall': final_row['final_recall'],
                'final_global_f1_score': final_row['final_f1_score'],
                
                # Métricas do filtro de clientes
                'filter_accuracy': filter_metrics['filter_accuracy'],
                'filter_precision': filter_metrics['filter_precision'],
                'filter_recall': filter_metrics['filter_recall'],
                'filter_f1_score': filter_metrics['filter_f1_score'],

                # Métricas da decisão de atualização
                'update_decision_accuracy': update_decision_metrics['update_decision_accuracy'],
                'update_decision_FN_count': update_decision_metrics['update_decision_false_negatives'], # O mais crítico!

                # Métricas de custo
                'median_aggregation_time_s': median_agg_time,
                'std_aggregation_time_s': std_agg_time,
                'total_experiment_time_s': total_time,
                'total_communication_mb': df_exp['communication_bytes'].sum() / (1024*1024),
            }
            final_results.append(exp_summary)

        # Cria o DataFrame final e salva em um novo arquivo CSV
        df_final_summary = pd.DataFrame(final_results)
        output_path = self.results_dir / 'final_metrics_summary.csv'
        df_final_summary.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"✅ Tabela de métricas finais salva com sucesso em: {output_path}")

    def create_all_plots(self):
        """
        Cria todos os plots de visualização a partir dos resultados carregados.
        """
        if not self.all_results:
            print("⚠️ Sem dados para plotar. Pulando a criação de gráficos.")
            return

        print("\n🎨 Iniciando a criação de todos os gráficos...")

        perf_data = []
        for result in self.all_results.values():
            if result.get('success'):
                common_info = {
                    'defense': result['defense'],
                    'attack_rate': result['attack_rate'],
                    'selection_fraction': result['selection_fraction'],
                    'final_accuracy': result.get('final_accuracy', 0.0),
                }
                perf_data.append(common_info)

        df_perf = pd.DataFrame(perf_data)
        
        defenses = sorted(df_perf['defense'].unique())
        colors = plt.colormaps.get('tab10').colors
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', '<', '>', 'X']
        defense_styles = {
            defense: {'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]}
            for i, defense in enumerate(defenses)
        }
        
        self._plot_accuracy_vs_attack(df_perf, defense_styles)
        self._plot_performance_heatmap(df_perf)
        self._plot_selection_impact(df_perf, defense_styles)
        self._plot_temporal_evolution(defense_styles)
        self._plot_cost_metrics(defense_styles)
        print("✅ Gráficos criados com sucesso.")

    def _plot_accuracy_vs_attack(self, df, defense_styles):
        """
        Plota a acurácia final vs. taxa de ataque para cada fração de seleção.
        """
        selection_fractions = sorted(df['selection_fraction'].unique())
        if not selection_fractions: return

        num_plots = len(selection_fractions)
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6), squeeze=False)
        axes = axes.flatten()

        for i, sel_frac in enumerate(selection_fractions):
            ax = axes[i]
            df_sub = df[df['selection_fraction'] == sel_frac]

            for defense, style in defense_styles.items():
                defense_data = df_sub[df_sub['defense'] == defense].sort_values('attack_rate')
                if not defense_data.empty:
                    ax.plot(defense_data['attack_rate'] * 100, defense_data['final_accuracy'],
                            label=defense, linewidth=2, markersize=7, alpha=0.9, **style)

            ax.set_xlabel('Taxa de Ataque (%)', fontsize=12)
            ax.set_ylabel('Acurácia Final', fontsize=12)
            ax.set_title(f'Desempenho (Seleção de {int(sel_frac*100)}% dos Clientes)', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.legend(title='Defesas', fontsize=10)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'accuracy_vs_attack_rate.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"💾 Gráfico de Acurácia vs Ataque salvo em: {plot_path}")

    def _plot_temporal_evolution(self, defense_styles):
        """
        Plota a evolução da acurácia ao longo dos rounds para diferentes cenários.
        """
        attack_rates = sorted(list({r['attack_rate'] for r in self.all_results.values() if r.get('success')}))
        selection_fractions = sorted(list({r['selection_fraction'] for r in self.all_results.values() if r.get('success')}))

        for sel_frac in selection_fractions:
            
            n_attacks = len(attack_rates)
            if n_attacks == 0: continue
            
            cols = min(3, n_attacks)
            rows = (n_attacks + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False, sharey=True)
            axes = axes.flatten()

            for i, attack_rate in enumerate(attack_rates):
                ax = axes[i]
                
                for exp_name, result in self.all_results.items():
                    if (result.get('success') and
                        result['attack_rate'] == attack_rate and
                        result['selection_fraction'] == sel_frac):
                        
                        metrics = result['metrics']
                        rounds = metrics.get('round', list(range(1, len(metrics.get('accuracy', [])) + 1))) # FLAG: Prioriza a chave 'round' se existir
                        accuracies = metrics.get('accuracy', [])
                        
                        if rounds and accuracies:
                            defense = result['defense']
                            style = defense_styles.get(defense, {})
                            ax.plot(rounds, accuracies, label=defense, linewidth=2, alpha=0.9, **style)

                ax.set_title(f"Ataque: {int(attack_rate*100)}%", fontsize=12, fontweight='bold')
                ax.set_xlabel('Round', fontsize=11)
                ax.set_ylabel('Acurácia', fontsize=11)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylim(0, 1.05)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 0), fontsize=11)
            
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f'Evolução da Acurácia (Seleção de {int(sel_frac*100)}%)', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            plot_path = self.plots_dir / f'temporal_evolution_sel_{int(sel_frac*100)}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Gráfico de Evolução Temporal salvo em: {plot_path}")
    
    def _plot_performance_heatmap(self, df):
        """Cria um heatmap de acurácia (Defesa vs. Taxa de Ataque) para cada fração de seleção."""
        selection_fractions = sorted(df['selection_fraction'].unique())
        
        for sel_frac in selection_fractions:
            df_sub = df[df['selection_fraction'] == sel_frac]
            if df_sub.empty: continue
            
            pivot_table = df_sub.pivot_table(
                index='defense', columns='attack_rate', values='final_accuracy'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table,
                annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Acurácia Final'}
            )
            plt.title(f'Heatmap de Acurácia (Seleção {int(sel_frac*100)}%)', fontsize=16, fontweight='bold')
            plt.xlabel('Taxa de Ataque (%)', fontsize=12)
            plt.xticks([i + 0.5 for i in range(len(pivot_table.columns))], [f'{int(c*100)}%' for c in pivot_table.columns])
            plt.ylabel('Defesa', fontsize=12)
            plt.tight_layout()
            
            plot_path = self.plots_dir / f'heatmap_performance_sel_{int(sel_frac*100)}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Gráfico de Heatmap salvo em: {plot_path}")

    def _plot_selection_impact(self, df, defense_styles):
        """Plota o impacto na acurácia ao mudar a fração de seleção de clientes."""
        selection_fractions = sorted(df['selection_fraction'].unique())
        
        if len(selection_fractions) != 2:
            print("⚠️ Gráfico de Impacto de Seleção pulado (requer exatamente 2 frações de seleção).")
            return
            
        frac_low, frac_high = selection_fractions[0], selection_fractions[1]
        
        pivot_low = df[df['selection_fraction'] == frac_low].pivot_table(index='defense', columns='attack_rate', values='final_accuracy')
        pivot_high = df[df['selection_fraction'] == frac_high].pivot_table(index='defense', columns='attack_rate', values='final_accuracy')
        
        # Alinha os dataframes e calcula a diferença, preenchendo com 0 se um valor faltar
        pivot_diff = pivot_high.subtract(pivot_low, fill_value=0)
        
        plt.figure(figsize=(10, 7))
        for defense in pivot_diff.index:
            style = defense_styles.get(defense, {})
            plt.plot(
                [c * 100 for c in pivot_diff.columns], pivot_diff.loc[defense], 
                label=defense, linewidth=2, markersize=7, alpha=0.9,
                marker=style.get('marker', 'o'), color=style.get('color')
            )
        
        plt.axhline(0, color='black', linestyle='--', alpha=0.7)
        plt.title('Impacto da Seleção de Clientes na Acurácia', fontsize=16, fontweight='bold')
        plt.xlabel('Taxa de Ataque (%)', fontsize=12)
        plt.ylabel(f'Melhora na Acurácia ({int(frac_high*100)}% - {int(frac_low*100)}%)', fontsize=12)
        plt.legend(title='Defesas')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = self.plots_dir / 'selection_impact.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Gráfico de Impacto da Seleção salvo em: {plot_path}")

    def _plot_cost_metrics(self, defense_styles):
        """
        Cria boxplots para métricas de custo a partir dos dados brutos em self.all_results.
        Cada boxplot mostra a distribuição dos valores de todos os rounds para cada defesa.
        """
        # 1. Preparar os dados em um formato "longo", ideal para o Seaborn
        plot_data = []
        for result in self.all_results.values():
            if not result.get('success'):
                continue

            metrics = result.get('metrics', {})
            base_info = {
                'defense': result['defense'],
                'selection_fraction': result['selection_fraction']
            }

            # Extrai cada valor individual de cada métrica de custo
            for metric_name in ['communication_bytes', 'aggregation_time', 'round_time', 'memory_usage']:
                if metric_name in metrics and metrics[metric_name]:
                    for value in metrics[metric_name]:
                        plot_data.append({**base_info, 'metric': metric_name, 'value': value})
        
        if not plot_data:
            print("⚠️ Métricas de custo não encontradas nos resultados. Gráficos de custo pulados.")
            return

        df = pd.DataFrame(plot_data)

        color_palette = {defense: style['color'] for defense, style in defense_styles.items()}

        # 2. Iterar sobre cada fração de seleção e criar os gráficos
        selection_fractions = sorted(df['selection_fraction'].unique())
        
        for sel_frac in selection_fractions:
            df_sel = df[df['selection_fraction'] == sel_frac]
            if df_sel.empty:
                continue

            # Mapeamento de métricas para títulos e conversões de unidade
            cost_metrics_map = {
                'communication_bytes': {'title': 'Comunicação por Round (MB)', 'divisor': 1024*1024, 'ylabel': 'MB'},
                'aggregation_time': {'title': 'Tempo de Agregação (s)', 'divisor': 1, 'ylabel': 'Segundos'},
                'round_time': {'title': 'Tempo Total por Round (s)', 'divisor': 1, 'ylabel': 'Segundos'},
                'memory_usage': {'title': 'Uso de Memória (MB)', 'divisor': 1, 'ylabel': 'MB'}
            }
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for i, (metric, props) in enumerate(cost_metrics_map.items()):
                ax = axes[i]
                metric_data = df_sel[df_sel['metric'] == metric].copy()
                
                if not metric_data.empty:
                    # Aplica a conversão de unidade
                    metric_data['value'] = metric_data['value'] / props['divisor']
                    
                    sns.boxplot(ax=ax, data=metric_data, x='defense', y='value', palette=color_palette, hue='defense')
                    ax.set_title(props['title'], fontsize=14, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel(props['ylabel'], fontsize=12)
                    ax.tick_params(axis='x', rotation=45, labelsize=11)
                    ax.grid(True, linestyle='--', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'Dados não disponíveis', ha='center', va='center', fontsize=12, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.suptitle(f'Análise de Custo Computacional (Seleção {int(sel_frac*100)}%)', fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            plot_path = self.plots_dir / f'cost_metrics_sel_{int(sel_frac*100)}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Gráfico de Métricas de Custo salvo em: {plot_path}")
    
    def run_full_analysis(self):
        """Executa o pipeline completo de análise: carregar dados e criar plots."""
        # O método retorna True se resultados foram carregados, False caso contrário.
        if self.load_results():
            self.create_all_plots()
        else:
            # A mensagem de "Nenhum resultado" já é impressa dentro de load_results.
            print("Análise interrompida.")


class ExperimentRunner:
    """
    Classe para gerar configurações e executar a simulação de experimentos.
    """
    def __init__(self, resume_dir=None):
        # self.defense_pipelines = {
        #     'var 1': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 7, 'std_dev_multiplier': 1.5, 'min_rounds_history': 7}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 7, 'std_dev_multiplier': 1.5, 'min_rounds_history': 7}}},
        #     # 'var 2': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 5, 'std_dev_multiplier': 1.5, 'min_rounds_history': 5}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 5, 'std_dev_multiplier': 1.5, 'min_rounds_history': 5}}},
        #     # 'var 3': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}}},
        #     # 'var 4': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 7, 'std_dev_multiplier': 1.7, 'min_rounds_history': 7}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 7, 'std_dev_multiplier': 1.7, 'min_rounds_history': 7}}},
        #     # 'var 5': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 5, 'std_dev_multiplier': 1.7, 'min_rounds_history': 5}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 5, 'std_dev_multiplier': 1.7, 'min_rounds_history': 5}}},
        #     # 'var 6': { 'client_filters': [{'name': 'L2_DIRECTIONAL_FILTER','params': {'window_size': 3, 'std_dev_multiplier': 1.7, 'min_rounds_history': 3}}],'aggregation_strategy': {'name': 'FED_AVG'},'global_model_filter': {'name': 'L2_GLOBAL_MODEL_FILTER','params': {'window_size': 3, 'std_dev_multiplier': 1.7, 'min_rounds_history': 3}}}
        # }
        self.defense_pipelines = {
            # --- GRUPO 1: Baselines (Sem Filtros de Cliente) ---
            # Objetivo: Medir a performance base e o efeito de agregadores robustos.

            # 'Média Ponderada (Baseline)': {
            #     'client_filters': [],
            #     'aggregation_strategy': {'name': 'FED_AVG', 'params': {}},
            #     'global_model_filter': None
            # },
            # 'Média Aparada': {
            #     'client_filters': [],
            #     'aggregation_strategy': {'name': 'TRIMMED_MEAN', 'params': {'trim_ratio': 0.4}},
            #     'global_model_filter': None
            # },

            # --- GRUPO 2: Filtros de Cliente + Agregação Padrão ---
            # Objetivo: Isolar e medir a eficácia de cada filtro de cliente.

            # 'Filtro Krum + Média Ponderada': {
            #     'client_filters': [{'name': 'KRUM', 'params': {}}],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': None
            # },
            # 'Filtro Multi-Krum + Média Ponderada': {
            #     'client_filters': [{'name': 'MULTI_KRUM', 'params': {}}],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': None
            # },
            # 'Filtro Clustering + Média Ponderada': {
            #     'client_filters': [{'name': 'CLUSTERING', 'params': {}}],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': None
            # },
            # 'Filtro L2 Direcional + Média Ponderada': {
            #     'client_filters': [{
            #         'name': 'L2_DIRECTIONAL_FILTER',
            #         'params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}
            #     }],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': None
            # },

            # --- GRUPO 3: Pipelines de Defesa em Múltiplas Camadas ---
            # Objetivo: Testar o efeito combinado de diferentes tipos de filtros e agregadores.

            # 'Média Ponderada + Filtro L2 Global': {
            #     'client_filters': [],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': {
            #         'name': 'L2_GLOBAL_MODEL_FILTER',
            #         'params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}
            #     }
            # },
            # 'Filtro L2 Direcional + Média Ponderada + Filtro L2 Global': {
            #     'client_filters': [{
            #         'name': 'L2_DIRECTIONAL_FILTER',
            #         'params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}
            #     }],
            #     'aggregation_strategy': {'name': 'FED_AVG'},
            #     'global_model_filter': {
            #         'name': 'L2_GLOBAL_MODEL_FILTER',
            #         'params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}
            #     }
            # },
            'Filtro Clustering + Média Ponderada + Filtro L2 Global': {
                'client_filters': [{'name': 'CLUSTERING', 'params': {}}],
                'aggregation_strategy': {'name': 'FED_AVG'},
                'global_model_filter': {
                    'name': 'L2_GLOBAL_MODEL_FILTER',
                    'params': {'window_size': 3, 'std_dev_multiplier': 1.5, 'min_rounds_history': 3}
                }
            },
        }
        self.attack_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
        self.selection_fractions = [0.4, 1.0]
        self.non_iid = [True, False]

        self.selection_strategy = 'random'
        self.model_name = 'CNN_SIGN'
        self.dataset_name = 'SIGN'
        self.rounds = 30
        self.num_clients = 10
        self.local_epochs = 2
        self.batch_size = 16
        self.seed = 42

        self.save_client_models = False
        self.save_server_intermediate_models = False

        if resume_dir:
            self.base_dir = Path(resume_dir)
            print(f"🔄 Retomando simulação no diretório: {self.base_dir}")
        else:
            self.base_dir = Path(f'simulations/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            print(f"🆕 Iniciando nova simulação em: {self.base_dir}")
        
        self.config_dir = self.base_dir / 'config'
        self.results_dir = self.base_dir / 'results'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.existing_exp_names = set()
        print(f"🚀 Runner iniciado. Saída será salva em: {self.base_dir}")

    def load_existing_results(self):
        """Varre o diretório de resultados para encontrar experimentos já concluídos."""
        print("\n🔍 Verificando experimentos já concluídos...")
        if not self.results_dir.exists() or not any(self.results_dir.iterdir()):
            print("✅ Nenhum resultado existente encontrado.")
            return

        analyzer = ResultAnalyzer(self.base_dir)
        if analyzer.load_results():
            self.existing_exp_names = set(analyzer.all_results.keys())
        print(f"✅ Encontrados {len(self.existing_exp_names)} resultados existentes.")

    def generate_all_configs(self):
        """Gera um dicionário com todas as configurações de experimentos planejados."""
        planned_configs = {}
        for defense_name, defense_pipeline in self.defense_pipelines.items():
            for attack_rate in self.attack_rates:
                for sel_frac in self.selection_fractions:
                    for non_iid in self.non_iid:
                        attack_part = f"lf_{int(attack_rate*100)}" if attack_rate > 0 else "no_attack"
                        defense_part = slugify(defense_name)
                        sel_part = f"sel_{int(sel_frac*100)}"
                        non_iid_part = 'non_iid' if non_iid else 'iid'
                        exp_name = f"{attack_part}_{defense_part}_{sel_part}_{non_iid_part}"

                        config = {
                            'experiment': {
                                'name': exp_name,
                                'defense_name': defense_name,
                                'description': f"{defense_name} vs {int(attack_rate*100)}% label flipping (sel: {int(sel_frac*100)}%)",
                                'seed': self.seed,
                                'rounds': self.rounds,
                                'output_dir': str(self.results_dir),
                                'log_level': 'info',
                                'save_client_models': self.save_client_models,
                                'save_server_intermediate_models': self.save_server_intermediate_models
                            },
                            'dataset': {'name': self.dataset_name, 'non_iid': non_iid},
                            'model': {'type': self.model_name, 'local_epochs': self.local_epochs, 'batch_size': self.batch_size},
                            'server': {
                                'type': 'standard',
                                'address': '0.0.0.0:8000',
                                'defense_pipeline': defense_pipeline,
                                'selection_strategy': self.selection_strategy,
                                'selection_fraction': sel_frac,
                                'evaluation_interval': 1,
                                'max_concurrent_clients': 3
                            },
                            'clients': {
                                'num_clients': self.num_clients,
                                'honest_client_type': 'standard',
                                'malicious_client_type': 'label_flipping' if attack_rate > 0 else None,
                                'malicious_percentage': attack_rate
                            }
                        }
                        planned_configs[exp_name] = config
        return planned_configs

    def estimate_time(self, configs):
        """Estima tempo total baseado em experiência anterior."""
        num_pending_experiments = len(configs)
        if num_pending_experiments == 0:
            return 0

        base_time_per_client_per_round = 6  # segundos (estimativa conservadora)

        avg_selection_fraction = np.mean(self.selection_fractions) if self.selection_fractions else 0
        avg_clients_per_round = self.num_clients * avg_selection_fraction
        avg_time_per_round = avg_clients_per_round * base_time_per_client_per_round
        avg_time_per_experiment = self.rounds * avg_time_per_round
        
        total_seconds = num_pending_experiments * avg_time_per_experiment
        hours = total_seconds / 3600
        
        print(f"⏱️  Estimativa de Tempo:")
        print(f"  - 📊 {num_pending_experiments} experimentos pendentes")
        print(f"  - 🔄 {self.rounds} rounds por experimento")
        print(f"  - ⏱️  ~{avg_time_per_experiment/60:.1f} min por experimento (em média)")
        print(f"  - 🕐 Tempo total estimado: {hours:.1f} horas ({hours/24:.1f} dias)")
        
        if hours > 8:
            print(f"  - ⚠️  A execução pode ser longa. Considere executar em partes ou usar uma máquina mais rápida.")
        
        return total_seconds
    
    def run_workflow(self):
        """Orquestra todo o fluxo: carrega, filtra, estima, executa e salva."""
        if not FLSimulator:
            print("❌ FLSimulator não está disponível. Não é possível executar experimentos.")
            return

        self.load_existing_results()
        
        planned_configs = self.generate_all_configs()

        print(f"{planned_configs.keys()}")  # DEBUG
        print(f"{self.existing_exp_names}")  # DEBUG
        
        pending_configs = {
            name: cfg for name, cfg in planned_configs.items()
            if name not in self.existing_exp_names
        }

        print(f"\n📊 Status da Simulação:")
        print(f"  - {len(planned_configs)} experimentos planejados.")
        print(f"  - {len(self.existing_exp_names)} experimentos já concluídos.")
        print(f"  - {len(pending_configs)} experimentos pendentes para execução.")

        if not pending_configs:
            print("\n🎉 Todos os experimentos já foram concluídos!")
        else:
            self.estimate_time(pending_configs)
            
            response = input(f"\n❓ Deseja continuar com a execução dos {len(pending_configs)} experimentos pendentes? (y/N): ").lower().strip()
            if response != 'y':
                print("❌ Execução cancelada pelo usuário.")
                return

            print(f"\n🚀 Iniciando execução de {len(pending_configs)} experimentos...")
            start_time = time.time()
            
            for i, (name, config_data) in enumerate(pending_configs.items()):
                print(f"\n{'='*60}\n🔬 Executando {i+1}/{len(pending_configs)}: {name}\n{'='*60}")
                
                config_path = self.config_dir / f"{name}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                
                try:
                    simulator = FLSimulator(str(config_path), use_threads=False)
                    simulator.run_simulation()

                    output_exp_dir = self.results_dir / name
                    malicious_file_path = output_exp_dir / 'malicious_clients.json'
                    with open(malicious_file_path, 'w') as f:
                        json.dump({'malicious_clients': simulator.malicious_indices}, f, indent=2)
                    print(f"💾 Índices de clientes maliciosos salvos em {malicious_file_path}")

                    print(f"✅ Sucesso: {name}")
                except Exception as e:
                    print(f"❌ ERRO em {name}: {e}")
            
            total_time = time.time() - start_time
            print(f"\n🎉 Execução dos pendentes concluída em {total_time/60:.2f} minutos.")

        self.save_results_summary()

    def save_results_summary(self):
        """Cria ou atualiza o arquivo de resumo JSON com todos os resultados."""
        print("\n💾 Salvando resumo final dos resultados...")
        analyzer = ResultAnalyzer(self.base_dir)
        analyzer.load_results() # Sempre varre para garantir que tem tudo

        if not analyzer.all_results:
            print("⚠️ Nenhum resultado encontrado para criar o resumo.")
            return

        summary_data = {
            'experiment_info': {
                'total_experiments': len(analyzer.all_results),
                'defenses': list(self.defense_pipelines.keys()),
                'attack_rates': self.attack_rates,
                'selection_fractions': self.selection_fractions,
                'rounds': self.rounds,
                'timestamp': datetime.now().isoformat()
            },
            'results': analyzer.all_results
        }
        
        summary_path = self.results_dir / 'complete_results.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"📄 Resumo atualizado salvo em: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Script para executar e/ou analisar experimentos de defesas em FL.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--run',
        action='store_true',
        help='Modo Execução: Inicia uma nova simulação do zero.'
    )
    group.add_argument(
        '--analyze_dir',
        type=str,
        metavar='PATH',
        help='Modo Análise: Caminho para uma simulação para rodar apenas a análise.\n'
             'Exemplo: --analyze_dir simulation_20250912_011458'
    )
    group.add_argument(
        '--resume_dir',
        type=str,
        metavar='PATH',
        help='Modo Execução (Retomar): Caminho para uma simulação para continuar a execução.\n'
             'Exemplo: --resume_dir simulation_20250912_011458'
    )
    args = parser.parse_args()

    if args.analyze_dir:
        print("--- MODO DE ANÁLISE ATIVADO ---")
        analyzer = ResultAnalyzer(args.analyze_dir)
        analyzer.run_full_analysis()
    else:
        print("--- MODO DE EXECUÇÃO ATIVADO ---")
        # Se resume_dir for fornecido, ele continua. Se for --run, resume_dir é None e inicia uma nova.
        runner = ExperimentRunner(resume_dir=args.resume_dir)
        runner.run_workflow()
        
        print("\n--- INICIANDO ANÁLISE DOS RESULTADOS FINAIS ---")
        analyzer = ResultAnalyzer(runner.base_dir)
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main()