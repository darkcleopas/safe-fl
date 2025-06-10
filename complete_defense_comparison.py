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
        self.defenses = ['FED_AVG', 'TRIMMED_MEAN', 'KRUM', 'MULTI_KRUM', 'CLUSTERING']
        self.attack_rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
        self.selection_fractions = [0.4, 1.0]
        
        # Configurações fixas
        self.rounds = 100
        self.num_clients = 15
        self.local_epochs = 5
        self.batch_size = 32
        self.save_client_models = False  # Salvar modelos dos clientes após cada round
        self.save_server_intermediate_models = False  # Salvar modelos intermediários do servidor

        # Diretórios
        self.config_dir = Path('config/defense_comparison')
        self.results_dir = Path('results_defense_comparison')
        self.plots_dir = Path('plots_defense_comparison')
        
        # Criar diretórios
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Armazenar resultados
        self.all_results = {}
        
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
        
        return config, exp_name
    
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
        base_time_per_round = 8  # segundos por round (estimativa conservadora)
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
        """Executa todos os experimentos."""
        
        start_time = time.time()
        completed = 0
        total = len(configs)
        
        print(f"\n🚀 Iniciando {total} experimentos...")
        
        for exp_name, config_info in configs.items():
            print(f"\n{'='*60}")
            print(f"Experimento {completed+1}/{total}: {exp_name}")
            print(f"{'='*60}")
            
            try:
                # Executar experimento
                simulator = FLSimulator(config_info['config_path'], use_threads=False)
                metrics = simulator.run_simulation()
                
                # Armazenar resultado
                self.all_results[exp_name] = {
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
                
                print(f"✅ Concluído! Acurácia final: {self.all_results[exp_name]['final_accuracy']:.4f}")
                print(f"⏱️ Progresso: {completed}/{total} ({100*completed/total:.1f}%)")
                print(f"🕐 Tempo restante estimado: {remaining/3600:.1f}h")
                
            except Exception as e:
                print(f"❌ Erro: {e}")
                self.all_results[exp_name] = {
                    **config_info,
                    'error': str(e),
                    'success': False
                }
                completed += 1
        
        total_time = time.time() - start_time
        print(f"\n🎉 Todos os experimentos concluídos!")
        print(f"⏱️ Tempo total: {total_time/3600:.2f} horas")
        
        return self.all_results
    
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
        
        return str(plot_path)
    
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
        """Executa análise completa."""
        
        print("🔬 ANÁLISE COMPLETA DE DEFESAS EM FEDERATED LEARNING")
        print("="*60)
        
        # 1. Gerar configurações
        configs = self.generate_all_configs()
        
        # 2. Estimar tempo
        self.estimate_time(configs)
        
        # 3. Confirmar execução
        response = input("\n❓ Continuar com os experimentos? (y/N): ").lower()
        if response != 'y':
            print("❌ Execução cancelada.")
            return
        
        # 4. Executar experimentos
        results = self.run_experiments(configs)
        
        # 5. Analisar pontos de ruptura
        breaking_points = self.analyze_breaking_points()
        
        # 6. Criar plots
        plot_path = self.create_comparison_plots()
        
        # 7. Salvar resultados
        results_path, csv_path = self.save_results_summary()
        
        # 8. Resumo final
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\n🎉 ANÁLISE CONCLUÍDA!")
        print(f"✅ {successful}/{len(results)} experimentos bem-sucedidos")
        print(f"📊 Plots salvos em: {plot_path}")
        print(f"💾 Resultados em: {results_path}")
        
        return results, breaking_points


def main():
    """Função principal."""
    runner = DefenseComparisonRunner()
    results, breaking_points = runner.run_complete_analysis()
    return results, breaking_points


if __name__ == "__main__":
    main()