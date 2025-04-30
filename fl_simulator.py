import datetime
import os
import threading
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import gc
import yaml
import math
import random

from src.clients.standard_client import FLClient as BaseFLClient
from src.servers.standard_server import FLServer

class FLSimulator:
    """
    Simulates Federated Learning in-memory without REST communication.
    """
    def __init__(self, config_path: str, use_threads: bool = False):
        """
        Initialize the simulator with configuration.
        
        Args:
            config_path: Path to the YAML config file
            use_threads: If True, uses threads for parallel simulation; otherwise, sequential
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        
        self.use_threads = use_threads
        self.experiment_config = self.config['experiment']
        self.model_config = self.config['model']
        self.dataset_config = self.config['dataset']
        self.clients_config = self.config['clients']
        self.server_config = self.config['server']
        
        # Setting up seed for reproducibility
        self.seed = self.experiment_config['seed']
        random.seed(self.seed)
        
        # Configura diretório de saída
        self.run_log = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.folder_name = f"{self.experiment_config['name']}" + f"_{self.run_log}"
        self.base_dir = os.path.join(
            self.experiment_config.get('output_dir', './results'),
            self.folder_name
        )
        os.makedirs(self.base_dir, exist_ok=True)

        # Configurar logging
        self.setup_logging()

        self.save_client_models = self.experiment_config.get('save_client_models', False)
        if self.save_client_models:
            self.client_models_dir = os.path.join(self.base_dir, 'client_models')
            os.makedirs(self.client_models_dir, exist_ok=True)
            self.logger.info(f"Modelos de clientes serão salvos em {self.client_models_dir}")

        # Inicializa o server
        self.server = self._create_server()

        # Inicializa os clients
        self.clients = self._create_clients()
    
    def setup_logging(self):
        """Configura o sistema de logging."""
        log_level = self.experiment_config.get('log_level', 'info').upper()
        
        # Cria diretório de logs dentro do diretório de resultados        
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configura o arquivo de log
        log_file = os.path.join(self.logs_dir, 'simulator.log')
        
        # Configura o logger raiz
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Cria o logger para este componente
        self.logger = logging.getLogger("FLSimulator")
        
        # Cria manipulador de arquivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Cria formatador
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Adiciona o manipulador ao logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Arquivo de log criado em {log_file}")
    
    def _create_server(self) -> FLServer:
        """Creates and initializes the server"""
        server = FLServer(self.config, self.base_dir)
        return server
    
    def _create_clients(self) -> Dict[int, BaseFLClient]:
        """Creates and initializes all clients"""
        clients = {}
        num_clients = self.clients_config['num_clients']
        malicious_percentage = self.clients_config['malicious_percentage']
        malicious_client_type = self.clients_config['malicious_client_type']
        honest_client_type = self.clients_config['honest_client_type']
        
        # Calculate number of malicious clients
        num_malicious = math.floor(num_clients * malicious_percentage)

        # Determine which clients are malicious
        self.malicious_indices = sorted(random.sample(range(1, num_clients + 1), num_malicious))
        self.logger.info(f"Malicious clients: {self.malicious_indices}")
        
        self.logger.info(f"Initializing {num_clients} clients ({num_malicious} malicious)")
        
        # Create clients
        for client_id in range(1, num_clients + 1):
            # Determine if this client is malicious
            is_malicious = client_id in self.malicious_indices
            
            client_type = malicious_client_type if is_malicious else honest_client_type
            try:
                # Importar classe do client
                client_module = __import__(f"src.clients.{client_type}_client", fromlist=['FLClient'])
                FLClient: BaseFLClient = client_module.FLClient
            except ImportError:
                raise ImportError(f"Client type '{client_type}' not found")
            
            client = FLClient(client_id, self.config, experiment_dir=self.base_dir)
            
            clients[client_id] = client

            if self.save_client_models:
                # Create a directory for the client model
                client_model_dir = os.path.join(self.client_models_dir, f'client_{client_id}')
                os.makedirs(client_model_dir, exist_ok=True)
        
        return clients
    
    def _train_client(self, client: BaseFLClient, global_weights: List[np.ndarray]) -> Tuple[List[np.ndarray], int, float, float]:
        """
        Trains a client with the current global weights
        
        Args:
            client: Client instance to train
            global_weights: Current global model weights
            
        Returns:
            Tuple of (updated_weights, num_examples, loss, accuracy)
        """
        # Set the weights in the client's model
        client.update_model_weights(global_weights)
        
        # Train the model
        weights, num_examples, loss, accuracy = client.train_model()
        
        return weights, num_examples, loss, accuracy
    
    def simulate_round(self, round_num: int):
        """
        Simulates a single round of federated learning
        
        Args:
            round_num: Current round number
        """              
        self.server.start_round()

        # Get current global weights
        global_weights = self.server.model.get_weights()

        selected_client_ids = self.server.selected_clients
                
        if self.use_threads:

            self.logger.info("Usando multi-threading para treinamento de clientes")

            # Multi-threaded training
            threads = []
            thread_results = {}
            
            # Create a lock for thread safety
            lock = threading.Lock()
            
            def train_client_thread(client_id):
                client = self.clients[client_id]
                weights, num_examples, loss, accuracy = self._train_client(client, global_weights)
                
                # Thread-safe update of results
                with lock:
                    thread_results[client_id] = (weights, num_examples, loss, accuracy)
            
            # Start a thread for each selected client
            for client_id in selected_client_ids:
                thread = threading.Thread(target=train_client_thread, args=(client_id,))
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Collect results from threads
            client_updates = thread_results
            
        else:
            # Sequential training
            self.logger.info("Usando treinamento sequencial para clientes")

            for client_id in selected_client_ids:
                client = self.clients[client_id]
                weights, num_examples, local_loss, local_accuracy = self._train_client(client, global_weights)

                # Armazenar atualização
                self.server.round_updates[client_id] = (weights, num_examples, local_loss, local_accuracy)

                # Salvar modelo do cliente se a flag estiver ativada
                if self.save_client_models:
                    try:
                        # Salvar modelo
                        client_model_dir = os.path.join(self.client_models_dir, f'client_{client_id}')
                        model_path = os.path.join(client_model_dir, f'model_round_{self.server.current_round}.h5')
                        client.model.save(model_path)
                        self.logger.info(f"Modelo do cliente {client_id} (rodada {self.server.current_round}) salvo em {model_path}")
                        
                        # Limpar memória
                        gc.collect()
                    except Exception as e:
                        self.logger.error(f"Erro ao salvar modelo do cliente {client_id}: {str(e)}")

                # Armazenar métricas locais
                if client_id not in self.server.metrics['local_losses']:
                    self.server.metrics['local_losses'][client_id] = []
                    self.server.metrics['local_accuracies'][client_id] = []
                    self.server.metrics['client_examples'][client_id] = []
                
                self.server.metrics['local_losses'][client_id].append(local_loss)
                self.server.metrics['local_accuracies'][client_id].append(local_accuracy)
                self.server.metrics['client_examples'][client_id].append(num_examples)
        
        self.server.check_round_completion()

        # Clean up to free memory
        gc.collect()
            
    
    def run_simulation(self):
        """Runs the complete federated learning simulation"""
        self.logger.info(f"Starting Federated Learning Environment")

        # Log experiment details
        self.logger.info(f"Config file: {self.config_path}")
        self.logger.info(f"Experiment: {self.experiment_config['name']}")
        self.logger.info(f"Seed: {self.seed}")
        self.logger.info(f"Dataset: {self.dataset_config['name']}")
        self.logger.info(f"Model: {self.model_config['type']}")
        self.logger.info(f"Number of clients: {self.clients_config['num_clients']}")
        self.logger.info(f"Total rounds: {self.experiment_config['rounds']}")
        # Log malicious client configuration
        if self.clients_config['malicious_percentage'] > 0:
            self.logger.info(f"Malicious client configuration:")
            self.logger.info(f"  Honest client type: {self.clients_config['honest_client_type']}")
            self.logger.info(f"  Malicious client type: {self.clients_config['malicious_client_type']}")
            self.logger.info(f"  Percentage of malicious clients: {self.clients_config['malicious_percentage'] * 100}%")
            self.logger.info(f"  Number of malicious clients: {self.clients_config['num_clients'] * self.clients_config['malicious_percentage']}")
            self.logger.info(f"  IDs of malicious clients: {sorted(self.malicious_indices)}")
        else:
            self.logger.info("No malicious clients")
            self.logger.info(f"Client type: {self.clients_config['honest_client_type']}")
        self.logger.info("")
        
        # Run for the specified number of rounds
        total_rounds = self.experiment_config['rounds']
        for round_num in range(1, total_rounds + 1):
            self.simulate_round(round_num)
                
        return self.server.metrics