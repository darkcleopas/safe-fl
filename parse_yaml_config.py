#!/usr/bin/env python3
"""
Script para extrair informações de um arquivo de configuração YAML para Federated Learning.
Usado pelo script run_local_experiment.sh para configuração automática.
Valida obrigatoriamente a presença de configurações críticas.
"""

import yaml
import sys
import json

class ConfigError(Exception):
    """Erro específico para problemas na configuração."""
    pass

def extract_config_values(config_path):
    """
    Extrai valores relevantes de um arquivo de configuração YAML e valida sua presença.
    Falha explicitamente se configurações obrigatórias estiverem ausentes.
    
    Args:
        config_path: Caminho para o arquivo de configuração YAML
    
    Returns:
        Dicionário com valores extraídos
    
    Raises:
        ConfigError: Se alguma configuração obrigatória estiver ausente
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verificar seções obrigatórias
        required_sections = ['clients', 'server', 'experiment', 'dataset', 'model']
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Seção obrigatória '{section}' não encontrada no arquivo de configuração")
        
        # Definir e verificar campos obrigatórios em cada seção
        required_fields = {
            'clients': ['num_clients', 'honest_client_type', 'malicious_percentage', 'malicious_client_type'],
            'server': ['address'],
            'experiment': ['name', 'rounds', 'seed'],
            'dataset': ['name'],
            'model': ['type']
        }
        
        # Verificar cada campo obrigatório
        for section, fields in required_fields.items():
            for field in fields:
                if field not in config[section]:
                    raise ConfigError(f"Campo obrigatório '{field}' não encontrado na seção '{section}'")
        
        # Verificar campos adicionais condicionalmente
        if 'malicious_percentage' in config['clients'] and config['clients']['malicious_percentage'] > 0:
            if 'malicious_client_type' not in config['clients']:
                raise ConfigError("Campo 'malicious_client_type' é obrigatório quando 'malicious_percentage' > 0")
                    
        # Extrair valores sem valores padrão
        extracted = {
            "num_clients": config["clients"]["num_clients"],
            "server_address": config["server"]["address"],
            "rounds": config["experiment"]["rounds"],
            "seed": config["experiment"]["seed"],
            "dataset": config["dataset"]["name"],
            "model": config["model"]["type"],
            "experiment_name": config["experiment"]["name"],
            "honest_client_type": config["clients"]["honest_client_type"],
            "malicious_percentage": config["clients"]["malicious_percentage"],
            "malicious_client_type": config["clients"]["malicious_client_type"]
        }
        
        # Extrair a porta do servidor
        if ":" in extracted["server_address"]:
            extracted["server_host"], extracted["server_port"] = extracted["server_address"].split(":")
        else:
            raise ConfigError("Endereço do servidor deve incluir porta no formato 'host:porta'")
        
        return extracted
    except ConfigError as e:
        print(f"Erro de configuração: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao processar arquivo de configuração: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python parse_yaml_config.py config_path", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    extracted_config = extract_config_values(config_path)
    
    # Imprimir como JSON para fácil processamento pelo script bash
    print(json.dumps(extracted_config))