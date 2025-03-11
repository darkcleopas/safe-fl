import os
import yaml
import logging
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    Classe para carregar e validar configurações YAML para o framework.
    """
    
    def __init__(self):
        """Inicializa o carregador de configurações."""
        self.logger = logging.getLogger("ConfigLoader")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carrega um arquivo de configuração YAML.
        
        Args:
            config_path: Caminho para o arquivo de configuração
            
        Returns:
            Dicionário contendo a configuração carregada
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.logger.info(f"Configuração carregada com sucesso de {config_path}")
            
            # Validar a configuração
            self._validate_config(config)
            
            return config
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Valida a configuração carregada.
        
        Args:
            config: Dicionário de configuração a ser validado
            
        Raises:
            ValueError: Se a configuração for inválida
        """
        # Verificar seções obrigatórias
        required_sections = ['experiment', 'dataset', 'server', 'model', 'clients']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")
        
        # Verificar campos obrigatórios em cada seção
        # Experimento
        experiment = config['experiment']
        if 'name' not in experiment:
            raise ValueError("Campo 'name' não encontrado na seção 'experiment'")
        if 'rounds' not in experiment:
            raise ValueError("Campo 'rounds' não encontrado na seção 'experiment'")
        
        # Dataset
        dataset = config['dataset']
        if 'name' not in dataset:
            raise ValueError("Campo 'name' não encontrado na seção 'dataset'")
        
        # Servidor
        server = config['server']
        if 'strategy' not in server:
            raise ValueError("Campo 'strategy' não encontrado na seção 'server'")
        
        # Modelo
        model = config['model']
        if 'type' not in model:
            raise ValueError("Campo 'type' não encontrado na seção 'model'")
        
        # Clientes
        clients = config['clients']
        if 'num_clients' not in clients:
            raise ValueError("Campo 'num_clients' não encontrado na seção 'clients'")
        
        self.logger.info("Configuração validada com sucesso")
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mescla duas configurações, onde o override_config substitui valores em base_config.
        
        Args:
            base_config: Configuração base
            override_config: Configuração de substituição
            
        Returns:
            Configuração mesclada
        """
        merged_config = base_config.copy()
        
        # Mesclar recursivamente
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key] = self.merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        
        return merged_config
    
    def get_scenario_config(self, base_config_path: str, scenario_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carrega uma configuração base e opcionalmente a mescla com uma configuração de cenário.
        
        Args:
            base_config_path: Caminho para a configuração base
            scenario_path: Caminho opcional para a configuração de cenário
            
        Returns:
            Configuração final
        """
        # Carregar configuração base
        base_config = self.load_config(base_config_path)
        
        # Se um cenário for especificado, carregá-lo e mesclá-lo com a configuração base
        if scenario_path and os.path.exists(scenario_path):
            scenario_config = self.load_config(scenario_path)
            final_config = self.merge_configs(base_config, scenario_config)
            self.logger.info(f"Configuração base mesclada com cenário de {scenario_path}")
            return final_config
        
        return base_config