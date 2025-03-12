import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


class FLMetricsVisualizer:
    """
    Visualizador de métricas para Federated Learning, focado em análise de segurança.
    """
    
    def __init__(self, metrics_data: Dict[str, Any], output_dir: str = "visualizations"):
        """
        Inicializa o visualizador com os dados de métricas.
        
        Args:
            metrics_data: Dicionário com métricas do treinamento
            output_dir: Diretório para salvar as visualizações
        """
        self.metrics = metrics_data
        self.output_dir = output_dir
        
        # Criar diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extrair dados importantes
        self.rounds = self.metrics.get('rounds', [])
        self.accuracy = self.metrics.get('accuracy', [])
        self.loss = self.metrics.get('loss', [])
        self.selected_clients = self.metrics.get('selected_clients', [])
        self.num_examples = self.metrics.get('num_examples', [])
        self.local_losses = self.metrics.get('local_losses', {})
        self.local_accuracies = self.metrics.get('local_accuracies', {})
        
        # Identificar todos os clientes
        self.all_clients = set()
        for clients in self.selected_clients:
            self.all_clients.update(clients)
        self.all_clients = sorted(list(self.all_clients))
        
        # Verificar se temos dados suficientes
        if not self.rounds:
            raise ValueError("Dados de métrica não contêm informações sobre rodadas")
    
    def plot_global_performance(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico mostrando a evolução da acurácia e perda global ao longo das rodadas.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot de acurácia
        ax1.plot(self.rounds, self.accuracy, 'o-', color='blue', label='Acurácia Global')
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Evolução do Modelo Global ao Longo das Rodadas')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot de perda
        ax2.plot(self.rounds, self.loss, 'o-', color='red', label='Perda Global')
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Perda')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'global_performance.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_client_accuracy_comparison(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico comparando a acurácia local de cada cliente ao longo das rodadas.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot para cada cliente
        for client_id, accuracies in self.local_accuracies.items():
            # Encontrar em quais rodadas este cliente participou
            participated_rounds = []
            for i, clients in enumerate(self.selected_clients):
                if int(client_id) in clients:
                    participated_rounds.append(self.rounds[i])
            
            # Verificar se temos o mesmo número de rodadas e acurácias
            if len(participated_rounds) == len(accuracies):
                ax.plot(participated_rounds, accuracies, 'o-', label=f'Cliente {client_id}')
            else:
                # Se não tivermos uma correspondência exata, assumimos que as acurácias estão em ordem
                ax.plot(self.rounds[:len(accuracies)], accuracies, 'o-', label=f'Cliente {client_id}')
        
        # Plot da acurácia global para comparação
        ax.plot(self.rounds, self.accuracy, 'o-', color='black', linewidth=2, label='Global')
        
        ax.set_xlabel('Rodada')
        ax.set_ylabel('Acurácia')
        ax.set_title('Comparação de Acurácia entre Clientes')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'client_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_client_contributions(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico de barras empilhadas mostrando a proporção de exemplos 
        contribuídos por cada cliente em cada rodada.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        # Preparar dados para o gráfico de barras empilhadas
        contribution_data = {}
        
        for round_idx, clients in enumerate(self.selected_clients):
            round_num = self.rounds[round_idx]
            contribution_data[round_num] = {}
            
            # Inicializar todas as contribuições como 0
            for client in self.all_clients:
                contribution_data[round_num][client] = 0
            
            # Calcular contribuição de exemplos por cliente
            # Aqui, vamos simplificar assumindo que todos os clientes contribuem igualmente
            # em uma rodada (já que não temos o número exato de exemplos por cliente)
            for client in clients:
                contribution_data[round_num][client] = 1  # Pode ser ajustado com dados reais
        
        # Converter para DataFrame
        contribution_df = pd.DataFrame(contribution_data).T
        
        # Normalizar para porcentagens
        contribution_df = contribution_df.div(contribution_df.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Criar barras empilhadas
        bottom = np.zeros(len(contribution_df))
        for client in contribution_df.columns:
            values = contribution_df[client].values
            ax.bar(contribution_df.index, values, bottom=bottom, label=f'Cliente {client}')
            bottom += values
        
        ax.set_xlabel('Rodada')
        ax.set_ylabel('Porcentagem de Contribuição (%)')
        ax.set_title('Distribuição de Contribuições dos Clientes por Rodada')
        ax.legend(title='Cliente', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'client_contributions.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_local_vs_global_divergence(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico mostrando a divergência entre métricas locais e globais.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        # Calcular divergências de acurácia
        divergence_data = {}
        
        for client_id, accuracies in self.local_accuracies.items():
            divergence_data[client_id] = []
            
            for i, acc in enumerate(accuracies):
                if i < len(self.accuracy):
                    # Divergência é a diferença entre acurácia local e global
                    divergence_data[client_id].append(acc - self.accuracy[i])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot para cada cliente
        for client_id, divergences in divergence_data.items():
            # Encontrar em quais rodadas este cliente participou
            participated_rounds = []
            for i, clients in enumerate(self.selected_clients):
                if int(client_id) in clients:
                    participated_rounds.append(self.rounds[i])
            
            # Verificar se temos o mesmo número de rodadas e divergências
            if len(participated_rounds) == len(divergences):
                ax.plot(participated_rounds, divergences, 'o-', label=f'Cliente {client_id}')
            else:
                # Se não tivermos uma correspondência exata, assumimos que as divergências estão em ordem
                ax.plot(self.rounds[:len(divergences)], divergences, 'o-', label=f'Cliente {client_id}')
        
        # Linha de referência em y=0 (sem divergência)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Rodada')
        ax.set_ylabel('Divergência de Acurácia (Local - Global)')
        ax.set_title('Divergência entre Acurácia Local e Global')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'local_vs_global_divergence.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_client_correlation_matrix(self, save_fig: bool = True) -> Optional[plt.Figure]:
        """
        Cria um heatmap de correlação entre os clientes com base em suas métricas.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib ou None se não houver dados suficientes
        """
        # Construir DataFrame com acurácias locais para cada cliente
        accuracy_data = {}
        
        for client_id, accuracies in self.local_accuracies.items():
            accuracy_data[f'Cliente {client_id}'] = accuracies
        
        # Verificar se temos dados suficientes
        if not accuracy_data or max(len(accs) for accs in accuracy_data.values()) < 2:
            print("Dados insuficientes para calcular a matriz de correlação")
            return None
        
        # Converter para DataFrame
        df = pd.DataFrame(accuracy_data)
        
        # Calcular correlação
        corr_matrix = df.corr()
        
        # Criar heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Matriz de Correlação entre Clientes (Acurácia)')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'client_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attack_detection_metrics(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria visualizações específicas para detecção de ataques.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        # Aqui podemos implementar métricas específicas de segurança, como:
        # - Desvio padrão das acurácias locais por rodada (alta variância pode indicar ataques)
        # - Taxa de melhoria global vs. contribuições locais
        
        # Calcular desvio padrão das acurácias locais por rodada
        std_devs = []
        rounds_with_data = []
        
        for round_idx, round_num in enumerate(self.rounds):
            round_accuracies = []
            
            for client_id, accuracies in self.local_accuracies.items():
                if round_idx < len(accuracies):
                    round_accuracies.append(accuracies[round_idx])
            
            if round_accuracies:
                std_devs.append(np.std(round_accuracies))
                rounds_with_data.append(round_num)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(rounds_with_data, std_devs, 'o-', color='purple')
        ax.set_xlabel('Rodada')
        ax.set_ylabel('Desvio Padrão da Acurácia Local')
        ax.set_title('Variabilidade de Desempenho entre Clientes (Potencial Indicador de Ataque)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar linha de referência para um limite hipotético de detecção
        if std_devs:
            threshold = np.mean(std_devs) * 1.5
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                      label='Limiar de Detecção (Hipotético)')
            ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'attack_detection_metrics.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_visualizations(self):
        """
        Gera e salva todas as visualizações disponíveis.
        """
        print(f"Gerando visualizações em {self.output_dir}...")
        
        self.plot_global_performance()
        print("✓ Gráfico de desempenho global gerado")
        
        self.plot_client_accuracy_comparison()
        print("✓ Gráfico de comparação de acurácia entre clientes gerado")
        
        self.plot_client_contributions()
        print("✓ Gráfico de contribuições dos clientes gerado")
        
        self.plot_local_vs_global_divergence()
        print("✓ Gráfico de divergência local vs. global gerado")
        
        if self.plot_client_correlation_matrix():
            print("✓ Matriz de correlação entre clientes gerada")
        
        self.plot_attack_detection_metrics()
        print("✓ Métricas de detecção de ataques geradas")
        
        print("Todas as visualizações foram geradas com sucesso!")


def load_metrics_from_json(json_path: str) -> Dict[str, Any]:
    """
    Carrega métricas de um arquivo JSON.
    
    Args:
        json_path: Caminho para o arquivo JSON
    
    Returns:
        Dicionário com as métricas carregadas
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Visualizador de métricas para Federated Learning')
    parser.add_argument('--metrics', type=str, required=True, help='Caminho para o arquivo JSON de métricas')
    parser.add_argument('--output', type=str, default='visualizations', help='Diretório para salvar as visualizações')
    
    args = parser.parse_args()
    
    # Carregar métricas
    metrics = load_metrics_from_json(args.metrics)
    
    # Criar visualizador
    visualizer = FLMetricsVisualizer(metrics, args.output)
    
    # Gerar todas as visualizações
    visualizer.generate_all_visualizations()


# if __name__ == "__main__":
main()