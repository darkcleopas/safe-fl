import os
import json
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# Função para ler todos os experimentos
def read_experiments(base_dir='results/experiments'):
    experiments = []
    
    # Encontrar todos os diretórios de experimentos
    experiment_dirs = [d for d in glob.glob(f"{base_dir}/*") if os.path.isdir(d)]
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        config_path = os.path.join(exp_dir, 'config.yaml')
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        
        # Verificar se os arquivos existem
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            print(f"Arquivos ausentes para o experimento {exp_name}, pulando.")
            continue
        
        # Ler as configurações do experimento
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Ler as métricas do experimento
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Adicionar experimento à lista
        experiments.append({
            'name': exp_name,
            'config': config,
            'metrics': metrics
        })
    
    return experiments

# Criar uma legenda informativa para cada experimento
def create_label(exp):
    config = exp['config']
    
    # Nome do experimento
    exp_name = config.get('experiment', {}).get('name', exp['name'])
    
    # Estratégia de ataque (se houver)
    malicious_type = config.get('clients', {}).get('malicious_client_type', 'none')
    malicious_percentage = config.get('clients', {}).get('malicious_percentage', 0)
    
    # Estratégia de defesa (se houver)
    defense_strategy = config.get('server', {}).get('aggregation_strategy', 'FEDAVG')
    
    # Se não há ataque (clientes maliciosos), simplificar a legenda
    if malicious_type == 'standard' or malicious_percentage == 0:
        return f"{exp_name} - {defense_strategy}"
    else:
        return f"{exp_name} - {defense_strategy} vs {malicious_type} ({int(malicious_percentage*100)}%)"

# Função para aplicar suavização às curvas (opcional)
def smooth_curve(y, window_size=11, poly_order=3):
    """Aplica o filtro Savitzky-Golay para suavizar a curva"""
    if len(y) < window_size:
        return y
    return savgol_filter(y, window_size, poly_order)

# Função para plotar as acurácias com melhor visualização
def plot_accuracies(experiments, 
                    smooth=True, 
                    window_size=31, 
                    max_rounds=None, 
                    selected_experiments=None,
                    line_styles=None,
                    subplots=False):
    
    if selected_experiments:
        # Filtrar apenas os experimentos selecionados
        filtered_exps = [exp for exp in experiments if any(name in exp['name'] for name in selected_experiments)]
    else:
        filtered_exps = experiments

    # Ordenar os experimentos por nome
    filtered_exps.sort(key=lambda x: x['name'])
    
    if not filtered_exps:
        print("Nenhum experimento encontrado com os filtros aplicados!")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_exps)))
    
    if not line_styles:
        line_styles = ['-'] * len(filtered_exps)
    
    if subplots:
        fig, axes = plt.subplots(len(filtered_exps), 1, figsize=(12, 4*len(filtered_exps)), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax] * len(filtered_exps)
    
    for i, exp in enumerate(filtered_exps):
        metrics = exp['metrics']
        rounds = metrics.get('rounds', [])
        accuracies = metrics.get('accuracy', [])
        
        # Limitar o número de rodadas, se solicitado
        if max_rounds and len(rounds) > max_rounds:
            rounds = rounds[:max_rounds]
            accuracies = accuracies[:max_rounds]
        
        # Suavizar a curva, se solicitado
        if smooth and len(accuracies) > 5:
            smoothed_acc = smooth_curve(accuracies, window_size)
        else:
            smoothed_acc = accuracies
        
        label = create_label(exp)
        
        # Plotar com linhas mais finas e sem marcadores para melhor visualização
        if subplots:
            ax = axes[i]
            ax.plot(rounds, smoothed_acc, linewidth=1.5, 
                    label=label, color=colors[i], linestyle=line_styles[i])
            ax.set_title(label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylabel('Acurácia', fontsize=10)
            
            # Apenas o último subplot mostra o eixo x
            if i == len(filtered_exps) - 1:
                ax.set_xlabel('Rodada', fontsize=12)
        else:
            axes[0].plot(rounds, smoothed_acc, linewidth=1.5, 
                       label=label, color=colors[i], linestyle=line_styles[i])
    
    if not subplots:
        axes[0].set_title('Acurácia do modelo ao longo das rodadas', fontsize=16)
        axes[0].set_xlabel('Rodada', fontsize=14)
        axes[0].set_ylabel('Acurácia', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.3)
        axes[0].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Nome do arquivo baseado nos parâmetros
    smooth_str = "_smooth" if smooth else ""
    max_str = f"_max{max_rounds}" if max_rounds else ""
    plt.savefig(f'accuracies_plot{smooth_str}{max_str}.png', dpi=300)
    
    # plt.show()

# Função para plotar as perdas (losses) com melhor visualização
def plot_losses(experiments, 
                smooth=True, 
                window_size=31, 
                max_rounds=None, 
                selected_experiments=None,
                line_styles=None,
                subplots=False):
    
    if selected_experiments:
        # Filtrar apenas os experimentos selecionados
        filtered_exps = [exp for exp in experiments if any(name in exp['name'] for name in selected_experiments)]
    else:
        filtered_exps = experiments
    
    # Ordenar os experimentos por nome
    filtered_exps.sort(key=lambda x: x['name'])
    
    if not filtered_exps:
        print("Nenhum experimento encontrado com os filtros aplicados!")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_exps)))
    
    if not line_styles:
        line_styles = ['-'] * len(filtered_exps)
    
    if subplots:
        fig, axes = plt.subplots(len(filtered_exps), 1, figsize=(12, 4*len(filtered_exps)), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax] * len(filtered_exps)
    
    for i, exp in enumerate(filtered_exps):
        metrics = exp['metrics']
        rounds = metrics.get('rounds', [])
        losses = metrics.get('loss', [])
        
        # Limitar o número de rodadas, se solicitado
        if max_rounds and len(rounds) > max_rounds:
            rounds = rounds[:max_rounds]
            losses = losses[:max_rounds]
        
        # Suavizar a curva, se solicitado
        if smooth and len(losses) > 5:
            smoothed_loss = smooth_curve(losses, window_size)
        else:
            smoothed_loss = losses
        
        label = create_label(exp)
        
        # Plotar com linhas mais finas e sem marcadores para melhor visualização
        if subplots:
            ax = axes[i]
            ax.plot(rounds, smoothed_loss, linewidth=1.5, 
                    label=label, color=colors[i], linestyle=line_styles[i])
            ax.set_title(label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylabel('Loss', fontsize=10)
            
            # Apenas o último subplot mostra o eixo x
            if i == len(filtered_exps) - 1:
                ax.set_xlabel('Rodada', fontsize=12)
        else:
            axes[0].plot(rounds, smoothed_loss, linewidth=1.5, 
                       label=label, color=colors[i], linestyle=line_styles[i])
    
    if not subplots:
        axes[0].set_title('Loss do modelo ao longo das rodadas', fontsize=16)
        axes[0].set_xlabel('Rodada', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.3)
        axes[0].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Nome do arquivo baseado nos parâmetros
    smooth_str = "_smooth" if smooth else ""
    max_str = f"_max{max_rounds}" if max_rounds else ""
    plt.savefig(f'losses_plot{smooth_str}{max_str}.png', dpi=300)
    
    # plt.show()

# Função para analisar as tendências nos experimentos
def analyze_experiments(experiments):
    """Analisa os resultados e fornece insights sobre os experimentos"""
    print("\n=== ANÁLISE DOS EXPERIMENTOS ===")
    
    # Ordenar experimentos por acurácia final
    experiments_by_acc = sorted(
        experiments, 
        key=lambda x: x['metrics'].get('accuracy', [-1])[-1] if x['metrics'].get('accuracy') else -1,
        reverse=True
    )
    
    print("\nExperimentos ordenados por acurácia final:")
    for i, exp in enumerate(experiments_by_acc):
        config = exp['config']
        metrics = exp['metrics']
        
        if not metrics.get('accuracy'):
            continue
            
        final_acc = metrics.get('accuracy', [-1])[-1]
        exp_name = config.get('experiment', {}).get('name', exp['name'])
        defense = config.get('server', {}).get('aggregation_strategy', 'N/A')
        attack = config.get('clients', {}).get('malicious_client_type', 'none')
        attack_pct = config.get('clients', {}).get('malicious_percentage', 0)
        
        print(f"{i+1}. {exp_name}: Acurácia={final_acc:.4f}, Defesa={defense}, " +
              f"Ataque={attack} ({int(attack_pct*100)}%)")
    
    # Analisar efeito das estratégias de defesa
    print("\nComparação entre estratégias de defesa:")
    defense_strategies = {}
    
    for exp in experiments:
        config = exp['config']
        metrics = exp['metrics']
        
        if not metrics.get('accuracy'):
            continue
            
        defense = config.get('server', {}).get('aggregation_strategy', 'N/A')
        final_acc = metrics.get('accuracy', [-1])[-1]
        
        if defense not in defense_strategies:
            defense_strategies[defense] = []
        
        defense_strategies[defense].append(final_acc)
    
    for defense, accs in defense_strategies.items():
        avg_acc = sum(accs) / len(accs) if accs else 0
        print(f"- {defense}: Acurácia média={avg_acc:.4f} (em {len(accs)} experimentos)")

# Função principal com mais opções
def main():
    print("Lendo experimentos...")
    experiments = read_experiments()
    
    if not experiments:
        print("Nenhum experimento encontrado!")
        return
    
    print(f"Encontrados {len(experiments)} experimentos")
    
    # Configurações para os gráficos
    smooth = True           # Suavizar as curvas
    window_size = 31        # Tamanho da janela para suavização
    max_rounds = None       # Limitar número de rodadas (None = todas)
    selected_exps = None    # Filtrar experimentos específicos (None = todos)
    use_subplots = False    # Usar subplots separados para cada experimento
    
    # Exemplo de como filtrar apenas alguns experimentos:
    # selected_exps = ['base', 'defense']  # Mostrar apenas experimentos com 'base' ou 'defense' no nome
    
    print("Gerando plot de acurácias...")
    plot_accuracies(
        experiments, 
        smooth=smooth, 
        window_size=window_size,
        max_rounds=max_rounds,
        selected_experiments=selected_exps,
        subplots=use_subplots
    )
    
    print("Gerando plot de losses...")
    plot_losses(
        experiments, 
        smooth=smooth, 
        window_size=window_size,
        max_rounds=max_rounds,
        selected_experiments=selected_exps,
        subplots=use_subplots
    )
    
    # Analisar os resultados
    analyze_experiments(experiments)
    
    # Nome dos arquivos baseado nos parâmetros
    smooth_str = "_smooth" if smooth else ""
    max_str = f"_max{max_rounds}" if max_rounds else ""
    print(f"\nPronto! Os plots foram salvos como 'accuracies_plot{smooth_str}{max_str}.png' e 'losses_plot{smooth_str}{max_str}.png'")

if __name__ == "__main__":
    main()