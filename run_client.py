import os
import yaml
import logging
import tensorflow as tf


logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limitar threads do TensorFlow
tf_num_threads = int(os.environ.get('TF_NUM_THREADS', '1'))
tf.config.threading.set_intra_op_parallelism_threads(tf_num_threads)
tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)


if __name__ == "__main__":
    # Carregar configuração
    config_path = os.environ.get('CONFIG_PATH', 'config/default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Obter ID do cliente do ambiente
    client_id = int(os.environ.get('CLIENT_ID', '1'))

    # Obter tipo de cliente do ambiente
    client_type = os.environ.get('CLIENT_TYPE', 'standard')
    
    # Obter URL do servidor do ambiente
    server_url = os.environ.get('SERVER_URL', f"http://{config['server']['address']}")

    try:
        # Importar classe do client
        server_module = __import__(f"src.clients.{client_type}_client", fromlist=['FLClient'])
        FLClient = server_module.FLClient
    except ImportError:
        raise ImportError(f"Server type '{client_type}' not found")
    
    # Inicializar e executar cliente
    client = FLClient(client_id, config, server_url)
    client.training_loop()
