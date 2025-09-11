import os
import json
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

BASE_DIR = 'experiment_results'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Criar diretório para os plots se não existir
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Função para ler todos os experimentos
def read_experiments():
    experiments = {}
    no_attack_exps = []  # Lista para armazenar experimentos sem ataque
    
    aggregation_dirs = [d for d in glob.glob(f"{BASE_DIR}/*") if os.path.isdir(d)]
    
    for aggregation_dir in aggregation_dirs:
        # Ignorar diretório de plots
        if os.path.basename(aggregation_dir) == 'plots':
            continue

        # Encontrar os diretórios com as variações de experimento
        exp_dirs = [d for d in glob.glob(f"{aggregation_dir}/*") if os.path.isdir(d)]
        if not exp_dirs:
            print(f"Nenhum experimento encontrado em {aggregation_dir}, pulando.")
            continue

        aggregation_name = os.path.basename(aggregation_dir)

        for exp_dir in exp_dirs:
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
            
            # Criar objeto de experimento
            experiment = {
                'name': exp_name,
                'config': config,
                'metrics': metrics,
                'aggregation': aggregation_name
            }
            
            # Extrair o tipo de ataque a partir do nome do experimento
            attack_type = extract_attack_type(exp_name)
            
            # Armazenar experimentos sem ataque separadamente
            if attack_type == 'no_attack':
                no_attack_exps.append(experiment)
            
            # Adicionar experimento ao dicionário
            if attack_type not in experiments:
                experiments[attack_type] = []
                
            experiments[attack_type].append(experiment)
    
    # Adicionar experimentos sem ataque a todos os outros tipos de ataque
    for attack_type, exps in experiments.items():
        if attack_type != 'no_attack':
            # Criar cópia dos experimentos sem ataque para cada tipo de ataque
            for no_attack_exp in no_attack_exps:
                # Verificar se já não existe um experimento sem ataque com o mesmo método de agregação
                if not any(exp['aggregation'] == no_attack_exp['aggregation'] and 'no_attack' in exp['name'] for exp in exps):
                    exps.append(no_attack_exp)
    
    return experiments

# Função para extrair o tipo de ataque do nome do experimento
def extract_attack_type(exp_name):
    # Exemplo de nomes: label_flipping_10_fed_avg, no_attack_trimmed_mean
    if exp_name.startswith('no_attack'):
        return 'no_attack'
    elif 'label_flipping' in exp_name:
        # Extrair o percentual de ataque (10, 20, 40, 80)
        parts = exp_name.split('_')
        if len(parts) >= 3:
            return f"label_flipping_{parts[2]}"
    
    # Caso não consiga identificar o tipo específico
    return 'unknown'

# Criar uma legenda informativa para cada experimento
def create_label(exp):
    aggregation_name = exp['aggregation']
    
    # Estratégia de ataque (se houver)
    malicious_type = exp['config'].get('clients', {}).get('malicious_client_type', 'none')
    malicious_percentage = exp['config'].get('clients', {}).get('malicious_percentage', 0)
    
    # Se não há ataque (clientes maliciosos), simplificar a legenda
    if malicious_type == 'standard' or malicious_percentage == 0:
        return f"{aggregation_name} - Sem Ataque"
    else:
        return f"{aggregation_name} vs {malicious_type} ({int(malicious_percentage*100)}%)"

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
                    attack_type=None,
                    line_styles=None,
                    subplots=False):
    
    # Ordenar os experimentos por nome, colocando experimentos sem ataque primeiro
    filtered_exps = sorted(experiments, key=lambda x: (0 if 'no_attack' in x['name'] else 1, x['name']))
    
    if not filtered_exps:
        print("Nenhum experimento encontrado com os filtros aplicados!")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_exps)))
    
    if not line_styles:
        # Usar linha sólida para experimentos sem ataque e linha tracejada para experimentos com ataque
        line_styles = []
        for exp in filtered_exps:
            if 'no_attack' in exp['name']:
                line_styles.append('-')  # Linha sólida
            else:
                line_styles.append('--')  # Linha tracejada
    
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
            ax.set_ylim(0, 1)  # Definir limite de 0 a 1 para acurácia
            
            # Apenas o último subplot mostra o eixo x
            if i == len(filtered_exps) - 1:
                ax.set_xlabel('Rodada', fontsize=12)
        else:
            axes[0].plot(rounds, smoothed_acc, linewidth=1.5, 
                       label=label, color=colors[i], linestyle=line_styles[i])
    
    if not subplots:
        if attack_type == 'no_attack':
            title = 'Acurácia do modelo sem ataques'
        else:
            title = f'Acurácia do modelo com ataque {attack_type}'
            
        axes[0].set_title(title, fontsize=16)
        axes[0].set_xlabel('Rodada', fontsize=14)
        axes[0].set_ylabel('Acurácia', fontsize=14)
        axes[0].set_ylim(0, 1)  # Definir limite de 0 a 1 para acurácia
        axes[0].grid(True, linestyle='--', alpha=0.3)
        axes[0].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Nome do arquivo baseado nos parâmetros
    smooth_str = "_smooth" if smooth else ""
    max_str = f"_max{max_rounds}" if max_rounds else ""
    attack_str = f"_{attack_type}" if attack_type else ""
    plt.savefig(f'{PLOTS_DIR}/accuracies_plot{attack_str}{smooth_str}{max_str}.png', dpi=300)
    
    # plt.show()

# Função para plotar as perdas (losses) com melhor visualização
def plot_losses(experiments, 
                smooth=True, 
                window_size=31, 
                max_rounds=None,
                attack_type=None,
                line_styles=None,
                subplots=False):
    
    # Ordenar os experimentos por nome, colocando experimentos sem ataque primeiro
    filtered_exps = sorted(experiments, key=lambda x: (0 if 'no_attack' in x['name'] else 1, x['name']))
    
    if not filtered_exps:
        print("Nenhum experimento encontrado com os filtros aplicados!")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_exps)))
    
    if not line_styles:
        # Usar linha sólida para experimentos sem ataque e linha tracejada para experimentos com ataque
        line_styles = []
        for exp in filtered_exps:
            if 'no_attack' in exp['name']:
                line_styles.append('-')  # Linha sólida
            else:
                line_styles.append('--')  # Linha tracejada
    
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
            ax.set_ylim(0, 25)  # Definir limite de 0 a 25 para loss
            
            # Apenas o último subplot mostra o eixo x
            if i == len(filtered_exps) - 1:
                ax.set_xlabel('Rodada', fontsize=12)
        else:
            axes[0].plot(rounds, smoothed_loss, linewidth=1.5, 
                       label=label, color=colors[i], linestyle=line_styles[i])
    
    if not subplots:
        if attack_type == 'no_attack':
            title = 'Loss do modelo sem ataques'
        else:
            title = f'Loss do modelo com ataque {attack_type}'
            
        axes[0].set_title(title, fontsize=16)
        axes[0].set_xlabel('Rodada', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].set_ylim(0, 25)  # Definir limite de 0 a 25 para loss
        axes[0].grid(True, linestyle='--', alpha=0.3)
        axes[0].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Nome do arquivo baseado nos parâmetros
    smooth_str = "_smooth" if smooth else ""
    max_str = f"_max{max_rounds}" if max_rounds else ""
    attack_str = f"_{attack_type}" if attack_type else ""
    plt.savefig(f'{PLOTS_DIR}/losses_plot{attack_str}{smooth_str}{max_str}.png', dpi=300)
    
    # plt.show()

# Função para analisar as tendências nos experimentos
def analyze_experiments(experiments, experiment_group=None):
    """Analisa os resultados e fornece insights sobre os experimentos"""
    print("\n=== ANÁLISE DOS EXPERIMENTOS ===")
    print(f"Grupo de experimentos: {experiment_group}")
    
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
    

# Função principal com mais opções
def main():
    print("Lendo experimentos...")
    experiments_by_attack = read_experiments()
    
    if not experiments_by_attack:
        print("Nenhum experimento encontrado!")
        return
    
    for attack_type, exps in experiments_by_attack.items():
        print(f"\nExperimentos para ataque {attack_type}:")
        for exp in exps:
            print(f" - {exp['name']} ({exp['aggregation']})")
        
    # Configurações para os gráficos
    smooth = True           # Suavizar as curvas
    window_size = 31        # Tamanho da janela para suavização
    max_rounds = None       # Limitar número de rodadas (None = todas)
    use_subplots = False    # Usar subplots separados para cada experimento
    
    print(f"Gerando plots na pasta {PLOTS_DIR}...")
    for attack_type, exps in experiments_by_attack.items():
        # Pular a categoria 'no_attack' uma vez que já está incluída em todas as outras
        if attack_type == 'no_attack':
            print(f"Gerando plots para experimentos sem ataque...")
        else:
            print(f"Gerando plots para ataque {attack_type}...")
        
        # # Plotar acurácias
        # plot_accuracies(
        #     exps, 
        #     smooth=smooth, 
        #     window_size=window_size,
        #     max_rounds=max_rounds,
        #     attack_type=attack_type,
        #     subplots=use_subplots
        # )
        
        # # Plotar losses
        # plot_losses(
        #     exps, 
        #     smooth=smooth, 
        #     window_size=window_size,
        #     max_rounds=max_rounds,
        #     attack_type=attack_type,
        #     subplots=use_subplots
        # )
        
        # Analisar os resultados
        analyze_experiments(exps, attack_type)
    
    print(f"\nPronto! Os gráficos foram salvos na pasta {PLOTS_DIR}")

if __name__ == "__main__":
    main()