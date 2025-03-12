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
        
        # Extrair dados de exemplos por cliente
        self.client_examples = self.metrics.get('client_examples', {})
        
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
        
        # Para cada rodada, coletamos os dados de contribuição de cada cliente
        for round_idx, round_num in enumerate(self.rounds):
            contribution_data[round_num] = {}
            
            # Inicializar contribuições como 0 para todos os clientes
            for client in self.all_clients:
                contribution_data[round_num][client] = 0
            
            # Encontrar quais clientes participaram desta rodada
            if round_idx < len(self.selected_clients):
                for client in self.selected_clients[round_idx]:
                    # Se temos dados de exemplos específicos para este cliente
                    client_str = str(client)
                    if client_str in self.client_examples:
                        # Verificar se temos dados para esta rodada específica
                        examples_list = self.client_examples[client_str]
                        if round_idx < len(examples_list):
                            contribution_data[round_num][client] = examples_list[round_idx]
                        else:
                            # Fallback: usar o primeiro valor disponível (ou outro método)
                            if examples_list:
                                contribution_data[round_num][client] = examples_list[0]
        
        # Converter para DataFrame
        contribution_df = pd.DataFrame(contribution_data).T
        
        # Criar figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Gráfico de contribuições absolutas
        bottom = np.zeros(len(contribution_df))
        for client in contribution_df.columns:
            values = contribution_df[client].values
            ax1.bar(contribution_df.index, values, bottom=bottom, label=f'Cliente {client}')
            bottom += values
        
        ax1.set_title('Número de Exemplos Contribuídos por Cliente em Cada Rodada')
        ax1.set_ylabel('Número de Exemplos')
        ax1.legend(title='Cliente', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 2. Gráfico de contribuições percentuais
        # Normalizar para porcentagens
        pct_df = contribution_df.div(contribution_df.sum(axis=1), axis=0) * 100
        
        bottom = np.zeros(len(pct_df))
        for client in pct_df.columns:
            values = pct_df[client].values
            ax2.bar(pct_df.index, values, bottom=bottom, label=f'Cliente {client}')
            bottom += values
        
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Porcentagem de Contribuição (%)')
        ax2.set_title('Distribuição Percentual de Contribuições dos Clientes por Rodada')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'client_contributions.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_examples_over_time(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico mostrando o número de exemplos contribuídos por cada cliente ao longo do tempo.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot para cada cliente
        for client_id, examples in self.client_examples.items():
            # Encontrar em quais rodadas este cliente participou
            participated_rounds = []
            for i, clients in enumerate(self.selected_clients):
                if int(client_id) in clients:
                    participated_rounds.append(self.rounds[i])
            
            # Verificar se temos o mesmo número de rodadas e exemplos
            if len(participated_rounds) == len(examples):
                ax.plot(participated_rounds, examples, 'o-', label=f'Cliente {client_id}')
            else:
                # Se não tivermos uma correspondência exata, assumimos que os exemplos estão em ordem
                rounds_to_plot = self.rounds[:len(examples)]
                if rounds_to_plot:
                    ax.plot(rounds_to_plot, examples, 'o-', label=f'Cliente {client_id}')
        
        ax.set_xlabel('Rodada')
        ax.set_ylabel('Número de Exemplos')
        ax.set_title('Evolução do Número de Exemplos por Cliente')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'examples_over_time.png'), dpi=300, bbox_inches='tight')
        
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
    
    def plot_examples_vs_performance(self, save_fig: bool = True) -> plt.Figure:
        """
        Cria um gráfico de dispersão comparando o número de exemplos com a acurácia por cliente.
        
        Args:
            save_fig: Se True, salva a figura no diretório de saída
        
        Returns:
            Objeto Figure do matplotlib
        """
        # Preparar dados para o gráfico
        plot_data = []
        
        for client_id in self.client_examples.keys():
            examples = self.client_examples[client_id]
            accuracies = self.local_accuracies.get(client_id, [])
            
            # Para cada rodada onde temos tanto exemplos quanto acurácia
            for i in range(min(len(examples), len(accuracies))):
                plot_data.append({
                    'Cliente': f'Cliente {client_id}',
                    'Exemplos': examples[i],
                    'Acurácia': accuracies[i],
                    'Rodada': i + 1 if i < len(self.rounds) else i
                })
        
        # Converter para DataFrame
        if not plot_data:
            print("Dados insuficientes para o gráfico de exemplos vs performance")
            return None
            
        df = pd.DataFrame(plot_data)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Usar seaborn para criar um scatterplot avançado
        scatter = sns.scatterplot(
            data=df,
            x='Exemplos',
            y='Acurácia',
            hue='Cliente',
            size='Rodada',
            sizes=(20, 200),
            alpha=0.7,
            ax=ax
        )
        
        # Adicionar rótulos
        for i, row in df.iterrows():
            ax.text(row['Exemplos'] + 0.1, row['Acurácia'], f"R{row['Rodada']}", 
                   fontsize=9, alpha=0.7)
        
        ax.set_title('Relação entre Número de Exemplos e Acurácia por Cliente')
        ax.set_xlabel('Número de Exemplos de Treinamento')
        ax.set_ylabel('Acurácia Local')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Ajustar legenda
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.output_dir, 'examples_vs_performance.png'), dpi=300, bbox_inches='tight')
        
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
        
        self.plot_examples_over_time()
        print("✓ Gráfico de evolução do número de exemplos por cliente gerado")
        
        self.plot_examples_vs_performance()
        print("✓ Gráfico de relação entre exemplos e performance gerado")
        
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


if __name__ == "__main__":
    main()