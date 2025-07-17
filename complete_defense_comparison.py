#!/usr/bin/env python3
"""
Script completo para comparação de defesas em Federated Learning.
Gera configurações, executa experimentos, analisa resultados e cria plots.
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
from datetime import datetime, timedelta
import numpy as np

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fl_simulator import FLSimulator


class DefenseComparisonRunner:
    """Runner para comparação completa de defesas."""
    
    def __init__(self):
        self.defenses = ['COSINE_SIMILARITY', 'FED_AVG', 'TRIMMED_MEAN', 'KRUM', 'MULTI_KRUM', 'CLUSTERING']
        self.attack_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
        self.selection_fractions = [0.4, 1.0]
        
        # Configurações fixas
        self.rounds = 50
        self.num_clients = 15
        self.local_epochs = 3
        self.batch_size = 64
        self.save_client_models = False  # Salvar modelos dos clientes após cada round
        self.save_server_intermediate_models = False  # Salvar modelos intermediários do servidor

        self.base_dir = Path(f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Diretórios
        self.config_dir = self.base_dir / 'config'
        self.results_dir = self.base_dir / 'results'
        self.plots_dir = self.base_dir / 'plots'
        
        # Criar diretórios
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Armazenar resultados
        self.all_results = {}
        self.existing_results = {}
        
    def create_config(self, defense, attack_rate, selection_fraction):
        """Cria configuração para um experimento específico."""
        
        # Nome do experimento
        attack_name = "no_attack" if attack_rate == 0.0 else f"label_flipping_{int(attack_rate*100)}"
        selection_name = "all_clients" if selection_fraction == 1.0 else f"sel_{int(selection_fraction*100)}"
        exp_name = f"{attack_name}_{defense.lower()}_{selection_name}"
        
        config = {
            'experiment': {
                'name': exp_name,
                'description': f"{defense} vs {int(attack_rate*100)}% label flipping (sel: {int(selection_fraction*100)}%)",
                'seed': 42,
                'rounds': self.rounds,
                'output_dir': str(self.results_dir),
                'log_level': 'info',
                'save_client_models': self.save_client_models,
                'save_server_intermediate_models': self.save_server_intermediate_models
            },
            'dataset': {
                'name': 'SIGN',
                'non_iid': True
            },
            'model': {
                'type': 'CNN_SIGN',
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size
            },
            'server': {
                'type': 'standard',
                'address': '0.0.0.0:8000',
                'aggregation_strategy': defense,
                'selection_strategy': 'random',
                'selection_fraction': selection_fraction,
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
        
        # Configurações específicas por defesa
        if defense == 'TRIMMED_MEAN':
            config['server']['trim_ratio'] = 0.4
        elif defense == 'COSINE_SIMILARITY':
            config['server'].update({
                'min_rounds': 10,
                'threshold': 0.4,
                'fallback_strategy': 'FED_AVG'
            })
        
        return config, exp_name
    
    def load_existing_results(self):
        """Carrega resultados existentes do diretório de resultados."""
        
        print("📊 Verificando experimentos já concluídos...")
        
        # Verificar se arquivo de resultados completos existe
        results_file = self.results_dir / 'complete_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    if 'results' in data:
                        self.existing_results = data['results']
                        print(f"✅ Carregados {len(self.existing_results)} experimentos existentes")
                        return self.existing_results
            except Exception as e:
                print(f"⚠️ Erro ao carregar arquivo de resultados: {e}")
        
        # Fallback: buscar por pastas individuais de experimentos
        existing_count = 0
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                metrics_file = exp_dir / 'metrics.json'
                config_file = exp_dir / 'config.yaml'
                
                if metrics_file.exists() and config_file.exists():
                    try:
                        # Carregar métricas
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        # Carregar configuração
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        # Extrair informações
                        exp_name = exp_dir.name
                        malicious_pct = config.get('clients', {}).get('malicious_percentage', 0.0)
                        selection_frac = config.get('server', {}).get('selection_fraction', 0.4)
                        defense = config.get('server', {}).get('aggregation_strategy', 'UNKNOWN')
                        
                        self.existing_results[exp_name] = {
                            'config_path': str(config_file),
                            'defense': defense,
                            'attack_rate': malicious_pct,
                            'selection_fraction': selection_frac,
                            'metrics': metrics,
                            'final_accuracy': metrics['accuracy'][-1] if metrics.get('accuracy') else 0.0,
                            'final_loss': metrics['loss'][-1] if metrics.get('loss') else float('inf'),
                            'convergence_rounds': len(metrics.get('accuracy', [])),
                            'success': True
                        }
                        existing_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Erro ao processar {exp_dir.name}: {e}")
        
        if existing_count > 0:
            print(f"✅ Encontrados {existing_count} experimentos através de busca por diretórios")
        else:
            print("📝 Nenhum experimento existente encontrado - começando do zero")
        
        return self.existing_results
    
    def filter_pending_configs(self, all_configs):
        """Filtra configurações que ainda precisam ser executadas."""
        
        existing_experiments = set(self.existing_results.keys())
        planned_experiments = set(all_configs.keys())
        
        # Experimentos que ainda precisam ser executados
        pending_experiments = planned_experiments - existing_experiments
        
        # Filtrar configurações pendentes
        pending_configs = {exp_name: all_configs[exp_name] 
                          for exp_name in pending_experiments}
        
        print(f"📈 Status dos experimentos:")
        print(f"  ✅ Já concluídos: {len(existing_experiments)}")
        print(f"  📝 Planejados: {len(planned_experiments)}")
        print(f"  ⏳ Pendentes: {len(pending_configs)}")
        
        if pending_configs:
            print(f"\n📋 Experimentos pendentes:")
            pending_by_defense = {}
            for exp_name, config_info in pending_configs.items():
                defense = config_info['defense']
                if defense not in pending_by_defense:
                    pending_by_defense[defense] = 0
                pending_by_defense[defense] += 1
            
            for defense, count in pending_by_defense.items():
                print(f"  🛡️ {defense}: {count} experimentos")
        
        return pending_configs
    
    def merge_results(self, new_results):
        """Mescla resultados existentes com novos resultados."""
        
        # Começar com resultados existentes
        merged_results = self.existing_results.copy()
        
        # Adicionar novos resultados
        merged_results.update(new_results)
        
        print(f"🔄 Mesclando resultados:")
        print(f"  📊 Existentes: {len(self.existing_results)}")
        print(f"  🆕 Novos: {len(new_results)}")
        print(f"  📈 Total: {len(merged_results)}")
        
        return merged_results
    
    def generate_all_configs(self):
        """Gera todas as configurações necessárias."""
        
        configs = {}
        total_experiments = len(self.defenses) * len(self.attack_rates) * len(self.selection_fractions)
        
        print(f"📝 Gerando {total_experiments} configurações...")
        
        for defense in self.defenses:
            for attack_rate in self.attack_rates:
                for selection_fraction in self.selection_fractions:
                    config, exp_name = self.create_config(defense, attack_rate, selection_fraction)
                    
                    # Salvar arquivo
                    config_path = self.config_dir / f'{exp_name}.yaml'
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    configs[exp_name] = {
                        'config_path': str(config_path),
                        'defense': defense,
                        'attack_rate': attack_rate,
                        'selection_fraction': selection_fraction
                    }
        
        print(f"✅ {len(configs)} configurações criadas em {self.config_dir}")
        return configs
    
    def estimate_time(self, configs):
        """Estima tempo total baseado em experiência anterior."""
        
        # Estimativas baseadas no número de rounds (linear)
        base_time_per_round = 6  # segundos por round (estimativa conservadora)
        time_per_round = 0
        num_clients_over_selection = [int(self.num_clients * s) for s in self.selection_fractions]
        for num_clients in num_clients_over_selection:
            time_per_round += base_time_per_round * num_clients 
        time_per_experiment = self.rounds * time_per_round
        total_time = (len(configs) / len(num_clients_over_selection)) * time_per_experiment
        
        # Converter para horas
        hours = total_time / 3600
        
        print(f"⏱️ Estimativa de Tempo:")
        print(f"  📊 {len(configs)} experimentos")
        print(f"  🔄 {self.rounds} rounds cada")
        print(f"  ⏱️ ~{time_per_experiment/60:.1f} min por experimento")
        print(f"  🕐 Tempo total estimado: {hours:.1f} horas ({hours/24:.1f} dias)")
        
        if hours > 8:
            print(f"  ⚠️ Considere executar em partes ou usar máquina mais rápida!")
        
        return total_time
    
    def run_experiments(self, configs):
        """Executa apenas os experimentos pendentes."""
        
        if not configs:
            print("✅ Todos os experimentos já foram concluídos!")
            return {}
        
        start_time = time.time()
        completed = 0
        total = len(configs)
        new_results = {}
        
        print(f"\n🚀 Iniciando {total} experimentos pendentes...")
        
        for exp_name, config_info in configs.items():
            print(f"\n{'='*60}")
            print(f"Experimento {completed+1}/{total}: {exp_name}")
            print(f"{'='*60}")
            
            try:
                # Executar experimento
                simulator = FLSimulator(config_info['config_path'], use_threads=False)
                metrics = simulator.run_simulation()
                
                # Armazenar resultado
                new_results[exp_name] = {
                    **config_info,
                    'metrics': metrics,
                    'final_accuracy': metrics['accuracy'][-1] if metrics['accuracy'] else 0.0,
                    'final_loss': metrics['loss'][-1] if metrics['loss'] else float('inf'),
                    'convergence_rounds': len(metrics['accuracy']),
                    'success': True
                }
                
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (total - completed) * avg_time
                
                print(f"✅ Concluído! Acurácia final: {new_results[exp_name]['final_accuracy']:.4f}")
                print(f"⏱️ Progresso: {completed}/{total} ({100*completed/total:.1f}%)")
                print(f"🕐 Tempo restante estimado: {remaining/3600:.1f}h")
                
            except Exception as e:
                print(f"❌ Erro: {e}")
                new_results[exp_name] = {
                    **config_info,
                    'error': str(e),
                    'success': False
                }
                completed += 1
        
        total_time = time.time() - start_time
        print(f"\n🎉 Experimentos pendentes concluídos!")
        print(f"⏱️ Tempo total: {total_time/3600:.2f} horas")
        
        return new_results
    
    def analyze_cost_metrics(self):
        """Analisa métricas de custo a partir dos resultados."""
        print("\n💰 Analisando métricas de custo...")

        cost_summary = []

        for exp_name, result in self.all_results.items():
            if not result.get('success'):
                continue

            metrics = result.get('metrics', {})
            communication = metrics.get('communication_bytes', [])
            aggregation = metrics.get('aggregation_time', [])
            round_time = metrics.get('round_time', [])
            memory = metrics.get('memory_usage', [])

            summary = {
                'experiment': exp_name,
                'defense': result['defense'],
                'attack_rate': result['attack_rate'],
                'selection_fraction': result['selection_fraction'],
                'avg_comm_bytes': np.mean(communication) if communication else 0,
                'avg_agg_time': np.mean(aggregation) if aggregation else 0,
                'avg_round_time': np.mean(round_time) if round_time else 0,
                'avg_memory': np.mean(memory) if memory else 0,
            }
            cost_summary.append(summary)

        df_cost = pd.DataFrame(cost_summary)
        cost_path = self.results_dir / 'cost_metrics_summary.csv'
        df_cost.to_csv(cost_path, index=False)

        print(f"✅ Métricas de custo salvas em: {cost_path}")
        return df_cost
    
    def analyze_breaking_points(self):
        """Analisa pontos de ruptura para cada defesa."""
        
        print("\n📊 Analisando pontos de ruptura...")
        
        breaking_points = {}
        accuracy_threshold = 0.5  # Limiar de falha
        
        for defense in self.defenses:
            breaking_points[defense] = {}
            
            for selection_fraction in self.selection_fractions:
                selection_name = "all_clients" if selection_fraction == 1.0 else f"sel_{int(selection_fraction*100)}"
                
                # Ordenar por taxa de ataque
                defense_results = []
                for attack_rate in sorted(self.attack_rates):
                    attack_name = "no_attack" if attack_rate == 0.0 else f"label_flipping_{int(attack_rate*100)}"
                    exp_name = f"{attack_name}_{defense.lower()}_{selection_name}"
                    
                    if exp_name in self.all_results and self.all_results[exp_name]['success']:
                        accuracy = self.all_results[exp_name]['final_accuracy']
                        defense_results.append((attack_rate, accuracy))
                
                # Encontrar ponto de ruptura
                breaking_point = None
                for attack_rate, accuracy in defense_results:
                    if accuracy < accuracy_threshold:
                        breaking_point = attack_rate
                        break
                
                if breaking_point is None:
                    breaking_point = 1.0  # Não quebrou nos nossos testes
                
                breaking_points[defense][selection_name] = {
                    'breaking_point': breaking_point,
                    'results': defense_results
                }
        
        # Imprimir resultados
        print("\n📈 Pontos de Ruptura (taxa de ataque onde acurácia < 0.5):")
        for defense in self.defenses:
            print(f"\n{defense}:")
            for selection_name, data in breaking_points[defense].items():
                bp = data['breaking_point']
                if bp < 1.0:
                    print(f"  {selection_name}: {bp*100:.0f}%")
                else:
                    print(f"  {selection_name}: >80% (não quebrou)")
        
        return breaking_points
    
    def create_comparison_plots(self):
        """Cria plots comparativos."""
        
        print("\n📊 Criando plots comparativos...")
        
        # Preparar dados para plots
        plot_data = []
        for exp_name, result in self.all_results.items():
            if result['success']:
                plot_data.append({
                    'defense': result['defense'],
                    'attack_rate': result['attack_rate'],
                    'selection_fraction': result['selection_fraction'],
                    'final_accuracy': result['final_accuracy'],
                    'final_loss': result['final_loss']
                })
        
        df = pd.DataFrame(plot_data)
        
        # Plot 1: Acurácia vs Taxa de Ataque por Defesa (Selection 40%)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        df_40 = df[df['selection_fraction'] == 0.4]
        for defense in self.defenses:
            defense_data = df_40[df_40['defense'] == defense]
            plt.plot(defense_data['attack_rate'] * 100, defense_data['final_accuracy'], 
                    'o-', label=defense, linewidth=2, markersize=6)
        
        plt.xlabel('Taxa de Ataque (%)')
        plt.ylabel('Acurácia Final')
        plt.title('Defesas vs Ataques Label Flipping (Seleção 40%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 2: Acurácia vs Taxa de Ataque por Defesa (Selection 100%)
        plt.subplot(2, 2, 2)
        df_100 = df[df['selection_fraction'] == 1.0]
        for defense in self.defenses:
            defense_data = df_100[df_100['defense'] == defense]
            plt.plot(defense_data['attack_rate'] * 100, defense_data['final_accuracy'], 
                    'o-', label=defense, linewidth=2, markersize=6)
        
        plt.xlabel('Taxa de Ataque (%)')
        plt.ylabel('Acurácia Final')
        plt.title('Defesas vs Ataques Label Flipping (Seleção 100%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 3: Comparação de Seleção de Clientes
        plt.subplot(2, 2, 3)
        attack_rates_plot = [0.2, 0.4, 0.6, 0.8]  # Ataques moderados a severos
        
        for defense in self.defenses:
            acc_40 = []
            acc_100 = []
            for rate in attack_rates_plot:
                acc_40.append(df_40[(df_40['defense'] == defense) & (df_40['attack_rate'] == rate)]['final_accuracy'].iloc[0] if len(df_40[(df_40['defense'] == defense) & (df_40['attack_rate'] == rate)]) > 0 else 0)
                acc_100.append(df_100[(df_100['defense'] == defense) & (df_100['attack_rate'] == rate)]['final_accuracy'].iloc[0] if len(df_100[(df_100['defense'] == defense) & (df_100['attack_rate'] == rate)]) > 0 else 0)
            
            improvement = [(a100 - a40) for a40, a100 in zip(acc_40, acc_100)]
            plt.plot(attack_rates_plot, improvement, 'o-', label=defense)
        
        plt.xlabel('Taxa de Ataque')
        plt.ylabel('Melhoria (100% - 40%)')
        plt.title('Impacto da Seleção de Clientes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Heatmap de Performance
        plt.subplot(2, 2, 4)
        
        # Criar matriz para heatmap
        heatmap_data = []
        for defense in self.defenses:
            row = []
            for attack_rate in self.attack_rates:
                # Usar seleção 40% para o heatmap
                accuracy = df_40[(df_40['defense'] == defense) & (df_40['attack_rate'] == attack_rate)]['final_accuracy']
                row.append(accuracy.iloc[0] if len(accuracy) > 0 else 0)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=[f'{int(r*100)}%' for r in self.attack_rates],
                   yticklabels=self.defenses,
                   cmap='RdYlGn',
                   vmin=0, vmax=1)
        plt.title('Acurácia por Defesa vs Taxa de Ataque')
        plt.xlabel('Taxa de Ataque')
        plt.ylabel('Defesa')
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'defense_comparison_complete.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Plot salvo em: {plot_path}")

        # Criar plots de evolução temporal
        self.create_temporal_evolution_plots()
        
        return str(plot_path)
    
    def create_temporal_evolution_plots(self):
        """Cria plots de evolução temporal da acurácia ao longo dos rounds."""
        
        print("\n📈 Criando plots de evolução temporal...")
        
        # Obter taxas de ataque únicas dos resultados (não usar constantes)
        available_attack_rates = set()
        for result in self.all_results.values():
            if result.get('success', False):
                available_attack_rates.add(result['attack_rate'])
        
        available_attack_rates = sorted(list(available_attack_rates))
        print(f"📊 Taxas de ataque encontradas: {[int(r*100) for r in available_attack_rates]}%")
        
        # Para cada fração de seleção, criar uma figura
        for selection_fraction in self.selection_fractions:
            selection_name = "100%" if selection_fraction == 1.0 else f"{int(selection_fraction*100)}%"
            
            # Configurar subplots baseado no número de taxas de ataque disponíveis
            n_attacks = len(available_attack_rates)
            cols = 3 if n_attacks > 3 else n_attacks
            rows = (n_attacks + cols - 1) // cols  # Ceil division
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()
            
            # Garantir que temos axes suficientes
            while len(axes) < n_attacks:
                axes.append(None)
            
            plot_count = 0
            
            for attack_rate in available_attack_rates:
                if plot_count >= len(axes) or axes[plot_count] is None:
                    break
                    
                ax = axes[plot_count]
                attack_name = "Sem Ataque" if attack_rate == 0.0 else f"{int(attack_rate*100)}% Atacantes"
                
                # Plotar cada defesa
                for defense in self.defenses:
                    line_style = '--' if defense == 'FED_AVG' else '-'
                    # Buscar experimento correspondente
                    attack_exp_name = "no_attack" if attack_rate == 0.0 else f"label_flipping_{int(attack_rate*100)}"
                    selection_exp_name = "all_clients" if selection_fraction == 1.0 else f"sel_{int(selection_fraction*100)}"
                    exp_name = f"{attack_exp_name}_{defense.lower()}_{selection_exp_name}"
                    
                    if exp_name in self.all_results and self.all_results[exp_name].get('success', False):
                        metrics = self.all_results[exp_name]['metrics']
                        rounds = metrics.get('rounds', [])
                        accuracies = metrics.get('accuracy', [])
                        
                        if rounds and accuracies:
                            ax.plot(rounds, accuracies, line_style, label=defense, 
                                   linewidth=2, markersize=4, alpha=0.8)
                
                ax.set_xlabel('Round')
                ax.set_ylabel('Acurácia')
                ax.set_title(attack_name)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                plot_count += 1
            
            # Ocultar subplots não utilizados
            for i in range(plot_count, len(axes)):
                if axes[i] is not None:
                    axes[i].set_visible(False)
            
            plt.suptitle(f'Evolução da Acurácia ao Longo dos Rounds (Seleção {selection_name})', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            
            # Salvar figura
            temporal_plot_path = self.plots_dir / f'temporal_evolution_selection_{int(selection_fraction*100)}.png'
            plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Plot temporal (seleção {selection_name}) salvo em: {temporal_plot_path}")
        
        return True
    
    def create_cost_plots(self, df_cost):
        """Cria plots das métricas de custo."""
        print("\n📉 Criando plots de custo...")

        plt.figure(figsize=(16, 10))

        metrics = ['avg_comm_bytes', 'avg_agg_time', 'avg_round_time', 'avg_memory']
        titles = ['Bytes Transmitidos', 'Tempo de Agregação (s)', 'Tempo por Round (s)', 'Uso de Memória (MB)']

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            sns.boxplot(data=df_cost, x='defense', y=metric)
            plt.title(titles[i])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        cost_plot_path = self.plots_dir / 'cost_comparison.png'
        plt.savefig(cost_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Plots de custo salvos em: {cost_plot_path}")

    def save_results_summary(self):
        """Salva resumo dos resultados."""
        
        # Criar resumo
        summary = {
            'experiment_info': {
                'total_experiments': len(self.all_results),
                'defenses': self.defenses,
                'attack_rates': self.attack_rates,
                'selection_fractions': self.selection_fractions,
                'rounds': self.rounds,
                'timestamp': datetime.now().isoformat()
            },
            'results': self.all_results
        }
        
        # Salvar JSON completo
        results_path = self.results_dir / 'complete_results.json'
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Criar CSV resumido
        csv_data = []
        for exp_name, result in self.all_results.items():
            if result['success']:
                csv_data.append({
                    'experiment': exp_name,
                    'defense': result['defense'],
                    'attack_rate': result['attack_rate'],
                    'selection_fraction': result['selection_fraction'],
                    'final_accuracy': result['final_accuracy'],
                    'final_loss': result['final_loss']
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.results_dir / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n💾 Resultados salvos:")
        print(f"  📊 JSON completo: {results_path}")
        print(f"  📈 CSV resumido: {csv_path}")
        
        return results_path, csv_path
    
    def run_complete_analysis(self):
        """Executa análise completa com suporte incremental."""
        
        print("🔬 ANÁLISE COMPLETA DE DEFESAS EM FEDERATED LEARNING")
        print("="*60)
        
        # 1. Carregar resultados existentes
        self.load_existing_results()
        
        # 2. Gerar todas as configurações (incluindo novas defesas)
        all_configs = self.generate_all_configs()
        
        # 3. Filtrar apenas configurações pendentes
        pending_configs = self.filter_pending_configs(all_configs)
        
        # 4. Estimar tempo apenas para experimentos pendentes
        self.estimate_time(pending_configs)
        
        # 5. Confirmar execução (só se houver experimentos pendentes)
        if pending_configs:
            response = input(f"\n❓ Continuar com {len(pending_configs)} experimentos pendentes? (y/N): ").lower()
            if response != 'y':
                print("❌ Execução cancelada.")
                # Ainda podemos fazer análise dos resultados existentes
                if self.existing_results:
                    print("📊 Fazendo análise apenas dos resultados existentes...")
                    self.all_results = self.existing_results
                else:
                    return
        
        # 6. Executar apenas experimentos pendentes
        if pending_configs:
            new_results = self.run_experiments(pending_configs)
            
            # 7. Mesclar resultados existentes + novos
            self.all_results = self.merge_results(new_results)
        else:
            # Usar apenas resultados existentes
            self.all_results = self.existing_results
        
        # 8. Analisar pontos de ruptura (dataset completo)
        breaking_points = self.analyze_breaking_points()
        
        # 9. Criar plots (dataset completo)
        plot_path = self.create_comparison_plots()
        
        # 10. Salvar resultados (dataset completo)
        results_path, csv_path = self.save_results_summary()

        # 11. Analisar métricas de custo
        df_cost = self.analyze_cost_metrics()
        self.create_cost_plots(df_cost)

        # 12. Resumo final
        successful = sum(1 for r in self.all_results.values() if r.get('success', False))
        total_planned = len(all_configs)
        
        print(f"\n🎉 ANÁLISE CONCLUÍDA!")
        print(f"✅ {successful}/{total_planned} experimentos bem-sucedidos")
        if pending_configs:
            print(f"🆕 {len(new_results)} novos experimentos executados")
        print(f"📊 Plots com dataset completo salvos em: {plot_path}")
        print(f"💾 Resultados completos em: {results_path}")
        
        return self.all_results, breaking_points


def main():
    """Função principal."""
    runner = DefenseComparisonRunner()
    results, breaking_points = runner.run_complete_analysis()
    return results, breaking_points


if __name__ == "__main__":
    main()