#!/usr/bin/env python3
"""
Script completo para comparação de defesas em Federated Learning.
Este script foi refatorado para separar a execução da análise.
- ExperimentRunner: Lida com a geração de configurações e execução de simulações.
- ResultAnalyzer: Lida com o carregamento e análise de resultados.
"""

import sys
import subprocess
import yaml
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
import re
import ast


# Adicionar diretório raiz ao path, se necessário
# (Assumindo que fl_simulator está no mesmo diretório ou no PYTHONPATH)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# A verificação de import do simulador foi movida para dentro do main (modo execução)


def slugify(text):
    """Converte um texto para um formato 'slug' seguro para nomes de arquivos."""
    text = text.lower()
    text = re.sub(r'\s+', '_', text)  # Substitui espaços por underscores
    text = re.sub(r'[^\w\-]+', '', text) # Remove caracteres não alfanuméricos
    return text


def ensure_list(x):
    """Garantir que o valor seja uma lista (para varrer variações)."""
    if x is None:
        return [None]
    return x if isinstance(x, list) else [x]


class ResultAnalyzer:
    """
    Classe para carregar e analisar resultados de experimentos.
    """
    def __init__(self, simulation_dir):
        """
        Inicializa o analisador com o diretório base da simulação.

        Args:
            simulation_dir (str or Path): O diretório da simulação contendo a pasta 'results'.
        """
        self.base_dir = Path(simulation_dir)
        self.results_dir = self.base_dir / 'results'

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
                            # Usa o nome da defesa do bloco experiment; fallback seguro
                            defense_name = config.get('experiment', {}).get('defense_name', 'Sem Defesa')

                            attack_rate = config.get('clients', {}).get('malicious_percentage', 0.0)
                            sel_frac = config.get('server', {}).get('selection_fraction', 0.0)
                            
                            # Listas de métricas podem estar vazias se a simulação falhou cedo
                            acc_list = metrics.get('accuracy') or []
                            loss_list = metrics.get('loss') or []
                            prec_list = metrics.get('precision') or []
                            rec_list = metrics.get('recall') or []
                            f1_list = metrics.get('f1') or []

                            # Determina se o experimento foi concluído com sucesso
                            total_rounds_config = config.get('experiment', {}).get('rounds')
                            rounds_executados = len(metrics.get('rounds', [])) if isinstance(metrics.get('rounds', []), list) else len(acc_list)
                            try:
                                total_rounds_config_int = int(total_rounds_config) if total_rounds_config is not None else None
                            except (TypeError, ValueError):
                                total_rounds_config_int = None
                            is_success = (total_rounds_config_int is not None) and (rounds_executados >= total_rounds_config_int)

                            self.all_results[exp_name] = {
                                'config_path': str(config_file),
                                'defense': defense_name,
                                'attack_rate': attack_rate,
                                'selection_fraction': sel_frac,
                                'selection_strategy': config.get('server', {}).get('selection_strategy'),
                                'dataset_name': config.get('dataset', {}).get('name'),
                                'non_iid': config.get('dataset', {}).get('non_iid'),
                                'dirichlet_alpha': config.get('dataset', {}).get('dirichlet_alpha'),
                                'model_type': config.get('model', {}).get('type'),
                                'local_epochs': config.get('model', {}).get('local_epochs'),
                                'batch_size': config.get('model', {}).get('batch_size'),
                                'learning_rate': config.get('model', {}).get('learning_rate'),
                                'rounds_total': config.get('experiment', {}).get('rounds'),
                                'num_clients': config.get('clients', {}).get('num_clients'),
                                'seed': config.get('experiment', {}).get('seed'),
                                'reuse_client_model': config.get('experiment', {}).get('reuse_client_model', False),
                                'keras_verbose': config.get('experiment', {}).get('keras_verbose', 0),
                                'malicious_clients': malicious_clients,
                                'metrics': metrics,
                                'final_accuracy': acc_list[-1] if acc_list else 0.0,
                                'final_loss': loss_list[-1] if loss_list else float('inf'),
                                'final_precision': prec_list[-1] if prec_list else 0.0,
                                'final_recall': rec_list[-1] if rec_list else 0.0,
                                'final_f1': f1_list[-1] if f1_list else 0.0,
                                'success': is_success,
                            }
                        except Exception as e:
                            print(f"❌ Erro ao processar o diretório {exp_dir.name}: {e}")
            print(f"✅ Varredura concluída. Encontrados {len(self.all_results)} resultados.")
        
        if not self.all_results:
             print("❌ Nenhum resultado para analisar.")
             return False
        return True
    
    def run_full_analysis(self):
        """Executa o pipeline completo de análise: carregar dados e processar."""
        # Evita revarrer se já está carregado
        if not self.all_results:
            if not self.load_results():
                print("Análise interrompida.")
                return

        if self.all_results:
            # NOVO: Instancia o processador para gerar DataFrame e gráficos
            processor = ResultProcessor(self.all_results, self.results_dir)
            processor.generate_and_save_dataframe()
            processor.generate_final_metrics_summary()
        else:
            print("Análise interrompida.")


class ResultProcessor:
    def __init__(self, all_results, results_dir):
        self.all_results = all_results
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

                try:
                    round_data = {
                        "experiment_id": exp_id,
                        "defense": exp_details.get("defense"),
                        "attack_rate": exp_details.get("attack_rate"),
                        "selection_fraction": exp_details.get("selection_fraction"),
                        "selection_strategy": exp_details.get("selection_strategy"),
                        "dataset_name": exp_details.get("dataset_name"),
                        "non_iid": exp_details.get("non_iid"),
                        "dirichlet_alpha": exp_details.get("dirichlet_alpha"),
                        "model_type": exp_details.get("model_type"),
                        "local_epochs": exp_details.get("local_epochs"),
                        "batch_size": exp_details.get("batch_size"),
                        "learning_rate": exp_details.get("learning_rate"),
                        "rounds_total": exp_details.get("rounds_total"),
                        "num_clients": exp_details.get("num_clients"),
                        "seed": exp_details.get("seed"),
                        "reuse_client_model": exp_details.get("reuse_client_model"),
                        "keras_verbose": exp_details.get("keras_verbose"),
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
                except (IndexError, TypeError, KeyError):
                    print(f"⚠️  Aviso: Pulando round {i} do exp {exp_id} devido a dados ausentes/corrompidos.")
                    continue

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
            total_time = df_exp['round_time'].dropna().sum()
            
            # Monta o dicionário com todos os resultados finais para este experimento
            exp_summary = {
                'experiment_id': exp_id,
                'defense': final_row['defense'],
                'attack_rate': final_row['attack_rate'],
                'selection_fraction': final_row['selection_fraction'],
                'selection_strategy': final_row.get('selection_strategy'),
                'dataset_name': final_row.get('dataset_name'),
                'non_iid': final_row.get('non_iid'),
                'dirichlet_alpha': final_row.get('dirichlet_alpha'),
                'model_type': final_row.get('model_type'),
                'local_epochs': final_row.get('local_epochs'),
                'batch_size': final_row.get('batch_size'),
                'learning_rate': final_row.get('learning_rate'),
                'rounds_total': final_row.get('rounds_total'),
                'num_clients': final_row.get('num_clients'),
                'seed': final_row.get('seed'),

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
                'total_communication_mb': df_exp['communication_bytes'].dropna().sum() / (1024*1024),
            }
            final_results.append(exp_summary)

        # Cria o DataFrame final e salva em um novo arquivo CSV
        df_final_summary = pd.DataFrame(final_results)
        output_path = self.results_dir / 'final_metrics_summary.csv'
        df_final_summary.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"✅ Tabela de métricas finais salva com sucesso em: {output_path}")



class ExperimentRunner:
    """
    Classe para gerar configurações e executar a simulação de experimentos.
    """
    def __init__(self, resume_dir=None):
        self.defense_pipelines = {
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
        # Defaults explícitos para evitar omissões silenciosas via suite
        self.learning_rate = 1e-4  # Pode ser lista no suite
        self.dirichlet_alpha = 0.1  # Pode ser lista no suite
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
            # Considera apenas experimentos marcados como sucesso
            self.existing_exp_names = {
                name for name, v in analyzer.all_results.items()
                if isinstance(v, dict) and v.get('success', False)
            }
        print(f"✅ Encontrados {len(self.existing_exp_names)} resultados existentes.")

    def collect_configs_from_directory(self):
        """Coleta todos os arquivos YAML em self.config_dir como fonte de verdade em retomadas.

        Returns:
            Dict[str, Path]: mapping de experiment_name -> caminho do YAML
        """
        configs = {}
        if not self.config_dir.exists():
            return configs
        for cfg_path in sorted(self.config_dir.glob('*.yaml')):
            try:
                exp_name = cfg_path.stem
                configs[exp_name] = cfg_path
            except Exception:
                # Pula qualquer arquivo estranho
                continue
        return configs

    def generate_all_configs(self):
        """Gera um dicionário com todas as configurações de experimentos planejados."""
        planned_configs = {}
        for defense_name, defense_pipeline in self.defense_pipelines.items():
            for selection_strategy in ensure_list(self.selection_strategy):
                for model_name in ensure_list(self.model_name):
                    for dataset_name in ensure_list(self.dataset_name):
                        for rounds in ensure_list(self.rounds):
                            for num_clients in ensure_list(self.num_clients):
                                for local_epochs in ensure_list(self.local_epochs):
                                    for batch_size in ensure_list(self.batch_size):
                                        lr_list = ensure_list(self.learning_rate) if self.learning_rate is not None else [None]
                                        for learning_rate in lr_list:
                                            for seed in ensure_list(self.seed):
                                                for attack_rate in ensure_list(self.attack_rates):
                                                    for sel_frac in ensure_list(self.selection_fractions):
                                                        for non_iid in ensure_list(self.non_iid):
                                                            # Se for IID, dirichlet_alpha não se aplica
                                                            if bool(non_iid):
                                                                alpha_list = ensure_list(self.dirichlet_alpha) if self.dirichlet_alpha is not None else [None]
                                                            else:
                                                                alpha_list = [None]
                                                            for dir_alpha in alpha_list:
                                                                # Nome do experimento contendo todas variações relevantes
                                                                attack_part = f"lf_{int(attack_rate*100)}" if attack_rate and attack_rate > 0 else "no_attack"
                                                                defense_part = slugify(defense_name)
                                                                sf_part = f"sf_{int(sel_frac*100)}"
                                                                strat_part = f"sel_{slugify(str(selection_strategy))}"
                                                                ds_part = f"ds_{slugify(str(dataset_name))}"
                                                                model_part = f"m_{slugify(str(model_name))}"
                                                                rounds_part = f"r_{rounds}"
                                                                clients_part = f"n_{num_clients}"
                                                                epochs_part = f"ep_{local_epochs}"
                                                                bs_part = f"bs_{batch_size}"
                                                                lr_part = f"lr_{learning_rate}" if learning_rate is not None else "lr_default"
                                                                seed_part = f"s_{seed}"
                                                                niid_part = 'non_iid' if non_iid else 'iid'
                                                                alpha_part = f"alpha_{dir_alpha}" if (bool(non_iid) and dir_alpha is not None) else None
                                                                name_parts = [attack_part, defense_part, sf_part, strat_part, ds_part, model_part, rounds_part, clients_part, epochs_part, bs_part, lr_part, seed_part, niid_part]
                                                                if alpha_part:
                                                                    name_parts.append(alpha_part)
                                                                exp_name = "_".join(name_parts)

                                                                config = {
                                                                    'experiment': {
                                                                        'name': exp_name,
                                                                        'defense_name': defense_name,
                                                                        'description': f"{defense_name} vs {int((attack_rate or 0)*100)}% label flipping (sel: {int(sel_frac*100)}%)",
                                                                        'seed': seed,
                                                                        'rounds': rounds,
                                                                        'output_dir': str(self.results_dir),
                                                                        'log_level': 'info',
                                                                        'save_client_models': self.save_client_models,
                                                                        'save_server_intermediate_models': self.save_server_intermediate_models,
                                                                        'reuse_client_model': getattr(self, 'reuse_client_model', False),
                                                                        'keras_verbose': int(getattr(self, 'keras_verbose', 0))
                                                                    },
                                                                    'dataset': {
                                                                        'name': dataset_name,
                                                                        'non_iid': bool(non_iid),
                                                                        'dirichlet_alpha': float(dir_alpha) if (bool(non_iid) and dir_alpha is not None) else None
                                                                    },
                                                                    'model': {
                                                                        'type': model_name,
                                                                        'local_epochs': int(local_epochs),
                                                                        'batch_size': int(batch_size)
                                                                    },
                                                                    'server': {
                                                                        'type': 'standard',
                                                                        'address': '0.0.0.0:8000',
                                                                        'defense_pipeline': defense_pipeline,
                                                                        'selection_strategy': selection_strategy,
                                                                        'selection_fraction': float(sel_frac),
                                                                        'evaluation_interval': 1,
                                                                        'max_concurrent_clients': int(getattr(self, 'max_concurrent_clients', 3))
                                                                    },
                                                                    'clients': {
                                                                        'num_clients': int(num_clients),
                                                                        'honest_client_type': 'standard',
                                                                        'malicious_client_type': 'label_flipping' if (attack_rate and attack_rate > 0) else None,
                                                                        'malicious_percentage': float(attack_rate or 0.0)
                                                                    }
                                                                }
                                                                if learning_rate is not None:
                                                                    config['model']['learning_rate'] = float(learning_rate)
                                                                planned_configs[exp_name] = config
        return planned_configs

    def estimate_time(self, configs):
        """Estima tempo total baseado em experiência anterior."""
        num_pending_experiments = len(configs)
        if num_pending_experiments == 0:
            return 0
        
        # Estimativas baseadas no tipo de modelo por round completo
        # O comentário "considerando 100 clientes" é a chave
        model_time_estimates_per_round = {
            'CNN_MNIST': 6,  # segundos para 100 clientes
            'DNN_MNIST': 2,  # segundos para 100 clientes
        }
        N_BASE_CLIENTS = 100.0  # Baseline de clientes para as estimativas acima
        DEFAULT_TIME = 6.0     # Tempo padrão se o modelo não for encontrado

        # 1. Calcular o tempo médio por cliente, com base nos modelos nos configs
        model_times_per_client = []
        for cfg in configs.values():
            model_type = cfg.get('model', {}).get('type')
            # Obter o tempo base (para 100 clientes) ou usar o padrão
            time_base_100_clients = model_time_estimates_per_round.get(model_type, DEFAULT_TIME)
            # Calcular o tempo por cliente individual
            time_per_client = time_base_100_clients / N_BASE_CLIENTS
            model_times_per_client.append(time_per_client)

        if not model_times_per_client:
            return 0  # Não deveria acontecer se num_pending_experiments > 0, mas é seguro

        avg_time_per_client_per_round = np.mean(model_times_per_client)

        # 2. Obter parâmetros médios da suíte (corrigindo bugs de 'None')
        # Preservando sua lógica 'ensure_list'
        avg_selection_fraction = float(np.mean(ensure_list(self.selection_fractions))) if self.selection_fractions is not None else 0.0
        avg_num_clients = float(np.mean(ensure_list(self.num_clients))) if self.num_clients is not None else 0.0
        avg_rounds = float(np.mean(ensure_list(self.rounds))) if self.rounds is not None else 0.0

        # 3. Calcular o tempo com base na nova lógica (corrigida)
        avg_clients_per_round = avg_num_clients * avg_selection_fraction
        
        # Esta é a nova lógica principal:
        avg_time_per_round = avg_clients_per_round * avg_time_per_client_per_round
        
        avg_time_per_experiment = (avg_rounds * avg_time_per_round) + 60  # Adiciona 60 segundos de overhead fixo por experimento
        
        total_seconds = num_pending_experiments * avg_time_per_experiment
        hours = total_seconds / 3600
        
        # 4. Imprimir o relatório
        print(f"⏱️  Estimativa de Tempo:")
        print(f"  - 📊 {num_pending_experiments} experimentos pendentes")
        print(f"  - 🔄 {int(avg_rounds)} rounds por experimento (média)")
        print(f"  - 👥 {avg_clients_per_round:.1f} clientes por round (média)")
        print(f"  - ⏱️  ~{avg_time_per_round:.2f} seg por round (média)")
        print(f"  - ⏱️  ~{avg_time_per_experiment/60:.1f} min por experimento (em média)")
        print(f"  - 🕐 Tempo total estimado: {hours:.1f} horas ({hours/24:.1f} dias)")
        
        if hours > 8:
            print(f"  - ⚠️  A execução pode ser longa. Considere executar em partes ou usar uma máquina mais rápida.")
        
        return total_seconds
    
    # run_workflow removido: fluxo de execução unificado no main usando simulate_fl.py

    def _derive_experiment_info(self, results: dict) -> dict:
        """Deriva metadados a partir dos resultados reais, evitando inconsistências."""
        defenses = sorted({v.get('defense') for v in results.values() if isinstance(v, dict)})
        attack_rates = sorted({float(v.get('attack_rate', 0.0)) for v in results.values() if isinstance(v, dict)})
        selection_fractions = sorted({float(v.get('selection_fraction', 0.0)) for v in results.values() if isinstance(v, dict)})
        rounds_candidates = [int(v.get('rounds_total')) for v in results.values() if isinstance(v, dict) and v.get('rounds_total')]
        rounds_val = max(rounds_candidates) if rounds_candidates else None
        return {
            'total_experiments': len(results),
            'defenses': defenses,
            'attack_rates': attack_rates,
            'selection_fractions': selection_fractions,
            'rounds': rounds_val,
            'timestamp': datetime.now().isoformat()
        }

    def save_results_summary(self, preloaded_results: dict | None = None):
        """Cria ou atualiza o arquivo de resumo JSON com todos os resultados.
        Se preloaded_results for fornecido, evita uma nova varredura do disco.
        """
        print("\n💾 Salvando resumo final dos resultados...")

        if preloaded_results is not None:
            results = preloaded_results
        else:
            analyzer = ResultAnalyzer(self.base_dir)
            analyzer.load_results()
            results = analyzer.all_results

        if not results:
            print("⚠️ Nenhum resultado encontrado para criar o resumo.")
            return

        experiment_info = self._derive_experiment_info(results)
        summary_data = {
            'experiment_info': experiment_info,
            'results': results
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
    parser.add_argument(
        '--suite',
        type=str,
        metavar='PATH',
        help='Arquivo YAML mestre descrevendo todos os experimentos (variações e defesas).'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Número de processos em paralelo para executar os experimentos (repasse para simulate_fl.py).'
    )
    parser.add_argument(
        '--threads',
        action='store_true',
        help='Usar multi-threading dentro de cada simulação (repasse para simulate_fl.py).'
    )
    parser.add_argument(
        '--tf-threads',
        type=int,
        default=None,
        help='Override do número de threads internos do TensorFlow por simulação (repasse para simulate_fl.py).'
    )
    args = parser.parse_args()

    if args.analyze_dir:
        print("--- MODO DE ANÁLISE ATIVADO ---")
        analyzer = ResultAnalyzer(args.analyze_dir)
        analyzer.run_full_analysis()
    else:
        # Verificação antecipada do simulador para evitar falhas tardias
        try:
            from fl_simulator import FLSimulator
            if FLSimulator is None:
                raise ImportError("FLSimulator é None")
        except Exception:
            print("❌ ERRO: Não foi possível importar FLSimulator.")
            print("   O modo de execução (--run) e retomada (--resume_dir) não funcionarão.")
            print("   Verifique sua instalação e o PYTHONPATH.")
            sys.exit(1)

        print("--- MODO DE EXECUÇÃO ATIVADO ---")
        # Se resume_dir for fornecido, ele continua. Se for --run, resume_dir é None e inicia uma nova.
        runner = ExperimentRunner(resume_dir=args.resume_dir)

        # Se um arquivo de suite for fornecido, sobrepõe as opções do runner
        # Agora permitido tanto em novas execuções quanto em retomadas: em retomadas,
        # as opções do suite serão usadas para GERAR NOVOS arquivos de config a serem
        # adicionados ao diretório existente, sem apagar os atuais.
        if args.suite:
            with open(args.suite, 'r') as f:
                suite_cfg = yaml.safe_load(f)

            # Atribui com defaults seguros
            runner.defense_pipelines = suite_cfg.get('defense_pipelines', runner.defense_pipelines)
            runner.attack_rates = suite_cfg.get('attack_rates', runner.attack_rates)
            runner.selection_fractions = suite_cfg.get('selection_fractions', runner.selection_fractions)
            runner.non_iid = suite_cfg.get('non_iid', runner.non_iid)
            runner.dirichlet_alpha = suite_cfg.get('dirichlet_alpha', runner.dirichlet_alpha)

            runner.selection_strategy = suite_cfg.get('selection_strategy', runner.selection_strategy)
            runner.model_name = suite_cfg.get('model_name', runner.model_name)
            runner.dataset_name = suite_cfg.get('dataset_name', runner.dataset_name)
            runner.rounds = suite_cfg.get('rounds', runner.rounds)
            runner.num_clients = suite_cfg.get('num_clients', runner.num_clients)
            runner.local_epochs = suite_cfg.get('local_epochs', runner.local_epochs)
            runner.batch_size = suite_cfg.get('batch_size', runner.batch_size)
            runner.learning_rate = suite_cfg.get('learning_rate', runner.learning_rate)
            runner.seed = suite_cfg.get('seed', runner.seed)

            runner.save_client_models = suite_cfg.get('save_client_models', runner.save_client_models)
            runner.save_server_intermediate_models = suite_cfg.get('save_server_intermediate_models', runner.save_server_intermediate_models)

            # Extras passados via experiment
            runner.reuse_client_model = suite_cfg.get('reuse_client_model', False)
            runner.keras_verbose = int(suite_cfg.get('keras_verbose', 0))
            runner.max_concurrent_clients = suite_cfg.get('max_concurrent_clients', 8)

        # Em retomadas, por padrão usamos os YAMLs já existentes na pasta config/ como fonte de verdade
        runner.load_existing_results()

        # Carrega resultados completos para identificar incompletos
        analyzer_for_pending = ResultAnalyzer(runner.base_dir)
        analyzer_for_pending.load_results()
        incomplete_exps = {
            name for name, v in analyzer_for_pending.all_results.items()
            if isinstance(v, dict) and not v.get('success', False)
        }

        if args.resume_dir:
            planned_configs_files = runner.collect_configs_from_directory()
            # Se não houver YAMLs no diretório, caímos para geração (retrocompatibilidade)
            if not planned_configs_files:
                print("⚠️ Nenhum YAML encontrado em config/. Gerando configurações a partir dos parâmetros do runner.")
                planned_configs = runner.generate_all_configs()
                # Persistir YAMLs gerados na pasta config/ para rastreabilidade
                for name, config_data in planned_configs.items():
                    config_path = runner.config_dir / f"{name}.yaml"
                    config_data['experiment']['reuse_client_model'] = getattr(runner, 'reuse_client_model', False)
                    config_data['experiment']['keras_verbose'] = int(getattr(runner, 'keras_verbose', 0))
                    config_data['server']['max_concurrent_clients'] = getattr(runner, 'max_concurrent_clients', 3)
                    with open(config_path, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                planned_configs_files = runner.collect_configs_from_directory()

            # NOVO: Se um suite foi passado junto com resume_dir, gere configs baseado no suite
            # e adicione apenas os que ainda não existem na pasta config/ atual.
            if args.suite:
                print("🧩 Mesclando novas configurações do suite na simulação existente...")
                suite_planned = runner.generate_all_configs()
                new_count = 0
                for name, config_data in suite_planned.items():
                    if name not in planned_configs_files:
                        config_path = runner.config_dir / f"{name}.yaml"
                        # Injetar flags extras de otimização na seção experiment/server
                        config_data['experiment']['reuse_client_model'] = getattr(runner, 'reuse_client_model', False)
                        config_data['experiment']['keras_verbose'] = int(getattr(runner, 'keras_verbose', 0))
                        config_data['server']['max_concurrent_clients'] = getattr(runner, 'max_concurrent_clients', 3)
                        with open(config_path, 'w') as f:
                            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                        planned_configs_files[name] = config_path
                        new_count += 1
                print(f"✅ {new_count} novas configs adicionadas a {runner.config_dir} a partir do suite.")

            # Calcula concluídos entre os planejados (com base nos resultados de sucesso)
            completed_among_planned = sum(1 for name in planned_configs_files.keys() if name in runner.existing_exp_names)

            # Pendentes: não concluídos com sucesso OU marcados como incompletos
            pending_files = {
                name: path for name, path in planned_configs_files.items()
                if (name not in runner.existing_exp_names) or (name in incomplete_exps)
            }

            print(f"\n📊 Status da Simulação:")
            print(f"  - {len(planned_configs_files)} configs encontrados em {runner.config_dir}.")
            print(f"  - {completed_among_planned} já concluídos com sucesso entre os planejados.")
            print(f"  - {len(pending_files)} pendentes para execução (inclui incompletos para reinício).")

            if pending_files:
                # Estima tempo usando apenas a quantidade de pendentes
                runner.estimate_time({k: {} for k in pending_files.keys()})

                response = input(f"\n❓ Deseja continuar com a execução dos {len(pending_files)} pendentes? (y/N): ").lower().strip()
                if response == 'y':
                    # Executa via simulate_fl.py apenas os arquivos pendentes (em lotes para evitar limite de linha de comando)
                    simulate_path = Path(__file__).resolve().parent / 'simulate_fl.py'
                    pending_list = list(pending_files.values())

                    # Define tamanho do lote conservador
                    batch_size = 64
                    for i in range(0, len(pending_list), batch_size):
                        batch = pending_list[i:i+batch_size]
                        cmd = [sys.executable, str(simulate_path), '--config'] + [str(p) for p in batch]
                        if args.threads:
                            cmd.append('--threads')
                        if args.tf_threads is not None:
                            cmd.extend(['--tf-threads', str(args.tf_threads)])
                        if args.jobs and args.jobs > 1:
                            cmd.extend(['--jobs', str(args.jobs)])
                        print(f"\n🧪 Executando lote {i//batch_size + 1}: {' '.join(cmd[:5])} ... (+{len(batch)} arquivos)")
                        subprocess.check_call(cmd)
                else:
                    print("❌ Execução cancelada pelo usuário.")
        else:
            # Fluxo original para novas execuções: gerar configs a partir dos parâmetros
            planned_configs = runner.generate_all_configs()
            pending_configs = {
                name: cfg for name, cfg in planned_configs.items()
                if name not in runner.existing_exp_names
            }

            print(f"\n📊 Status da Simulação:")
            print(f"  - {len(planned_configs)} experimentos planejados.")
            print(f"  - {len(runner.existing_exp_names)} experimentos já concluídos.")
            print(f"  - {len(pending_configs)} experimentos pendentes para execução.")

            if pending_configs:
                runner.estimate_time(pending_configs)

                response = input(f"\n❓ Deseja continuar com a execução dos {len(pending_configs)} experimentos pendentes? (y/N): ").lower().strip()
                if response == 'y':
                    # Escreve todos os YAMLs pendentes
                    for name, config_data in pending_configs.items():
                        config_path = runner.config_dir / f"{name}.yaml"
                        # Injetar flags extras de otimização na seção experiment/server
                        config_data['experiment']['reuse_client_model'] = getattr(runner, 'reuse_client_model', False)
                        config_data['experiment']['keras_verbose'] = int(getattr(runner, 'keras_verbose', 0))
                        config_data['server']['max_concurrent_clients'] = getattr(runner, 'max_concurrent_clients', 3)
                        with open(config_path, 'w') as f:
                            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

                    # Executa via simulate_fl.py com paralelismo em processos
                    simulate_path = Path(__file__).resolve().parent / 'simulate_fl.py'
                    cmd = [sys.executable, str(simulate_path), '--config', str(runner.config_dir)]
                    if args.threads:
                        cmd.append('--threads')
                    if args.tf_threads is not None:
                        cmd.extend(['--tf-threads', str(args.tf_threads)])
                    if args.jobs and args.jobs > 1:
                        cmd.extend(['--jobs', str(args.jobs)])
                    print(f"\n🧪 Executando: {' '.join(cmd)}")
                    subprocess.check_call(cmd)
                else:
                    print("❌ Execução cancelada pelo usuário.")

        # Seja tendo ou não executado algo, varre UMA vez e reaproveita nos passos seguintes
        analyzer = ResultAnalyzer(runner.base_dir)
        analyzer.load_results()
        runner.save_results_summary(preloaded_results=analyzer.all_results)

        print("\n--- INICIANDO ANÁLISE DOS RESULTADOS FINAIS ---")
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main()