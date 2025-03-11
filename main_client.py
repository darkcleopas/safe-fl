import os
import yaml
import logging

from src.clients.standard_client import FLClient


logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


if __name__ == "__main__":
    # Carregar configuração
    config_path = os.environ.get('CONFIG_PATH', 'config/default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Obter ID do cliente do ambiente
    client_id = int(os.environ.get('CLIENT_ID', '1'))
    
    # Obter URL do servidor do ambiente
    server_url = os.environ.get('SERVER_URL', f"http://{config['server']['address']}")
    
    # Inicializar e executar cliente
    client = FLClient(client_id, config, server_url)
    client.training_loop()