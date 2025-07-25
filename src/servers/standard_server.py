import os
import yaml
import numpy as np
import logging
import datetime
import json
import tensorflow as tf
from typing import List, Dict, Any, Tuple
import base64
import io
import gc
import inspect
import time
import psutil

from src.utils.aggregation_factory import AggregationFactory
from src.utils.dataset_factory import DatasetFactory
from src.utils.model_factory import ModelFactory


class FLServer:
    """
    Servidor base para Federated Learning.
    """
    def __init__(self, config: Dict[str, Any], base_dir: str = None):
        """
        Inicializa o servidor com a configuração fornecida.
        
        Args:
            config: Dicionário com a configuração do servidor
        """
        self.config = config
        
        # Configurações gerais
        self.experiment_config = config['experiment']
        self.model_config = config['model']
        self.server_config = config['server']
        self.dataset_config = config['dataset']
        self.clients_config = config['clients']

        # Configurar diretório de saída
        if base_dir:
            self.base_dir = base_dir
        else:
            self.run_log = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.folder_name = f"{self.experiment_config['name']}" + f"_{self.run_log}"
            self.base_dir = os.path.join(
                self.experiment_config.get('output_dir', './results'),
                self.folder_name
            )
            os.makedirs(self.base_dir, exist_ok=True)

        self.save_intermediate_server_models = self.experiment_config.get('save_intermediate_server_models', False)

        # Gerenciamento de clientes para controle de recursos
        self.max_concurrent_clients = config.get('server', {}).get('max_concurrent_clients', 2)
        self.client_queue = []
        self.active_clients = []
        
        # Inicializar estratégia de agregação
        aggregation_strategy_type = self.server_config.get('aggregation_strategy', 'FED_AVG')
        self.aggregation_strategy = AggregationFactory.create_aggregation_strategy(
            aggregation_strategy_type, 
            self.server_config
        )
        
        # Definir seed para reprodutibilidade
        self.seed = self.experiment_config.get('seed', 42)
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Configurar diretório de modelos intermediários
        if self.save_intermediate_server_models:
            self.intermediate_server_models_dir = os.path.join(self.base_dir, 'intermediate_server_models')
            os.makedirs(self.intermediate_server_models_dir, exist_ok=True)
        
        # Configurar logging
        self.setup_logging()
        
        # Salvar configuração
        self.save_config()
        
        # Métricas para acompanhamento
        self.metrics = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'selected_clients': [],
            'num_examples': [],
            'local_losses': {},
            'local_accuracies': {},
            'client_examples': {},
            'aggregation_time': [],  # Tempo de agregação por rodada
            'round_time': [],  # Tempo total por rodada
            'model_size_bytes': None,  # Tamanho do modelo em bytes
            'num_parameters': None,  # Número de parâmetros
            'communication_bytes': [],  # Bytes transmitidos por rodada
            'client_submission_times': {},  # Tempo de submissão por cliente
            'memory_usage': [],  # Uso de memória durante agregação
        }

        # Inicializar o modelo global
        self.model = None
        self.initialize_model()
        
        # Estado da simulação
        self.current_round = 0
        self.total_rounds = self.experiment_config['rounds']
        self.round_updates = {}  # Armazena atualizações por round
        self.selected_clients = []  # Clientes selecionados para o round atual
        self.training_complete = False # Indica se o treinamento foi concluído
        self.round_start_time = None
        self.client_start_times = {}

        self.logger.info(f"Servidor inicializado com sucesso usando estratégia de agregação {aggregation_strategy_type}")
    
    def setup_logging(self):
        """Configura o sistema de logging."""
        log_level = self.experiment_config.get('log_level', 'info').upper()
        
        # Cria diretório de logs dentro do diretório de resultados
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configura o manipulador de arquivo
        log_file = os.path.join(self.logs_dir, 'server.log')
        
        # Configura o logger raiz
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Cria o logger para este componente
        self.logger = logging.getLogger("FLServer")
        
        # Cria manipulador de arquivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Cria formatador
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Adiciona o manipulador ao logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Arquivo de log criado em {log_file}")
    
    def save_config(self):
        """Salva a configuração atual em arquivo."""
        config_path = os.path.join(self.base_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.info(f"Configuração salva em {config_path}")
    
    def initialize_model(self):
        """Inicializa o modelo global baseado na configuração."""
        
        dataset_factory = DatasetFactory()
        _, _, x_test, y_test, num_classes = dataset_factory.load_dataset(
            dataset_name=self.dataset_config['name'],
            client_id=0,  # Servidor tem ID 0
            num_clients=self.clients_config['num_clients'],
            non_iid=False,
            seed=self.seed,
            split="test"
        )
        
        self.x_test = x_test
        self.y_test = y_test
        
        # Inicializar o modelo
        model_factory = ModelFactory()
        self.model = model_factory.create_model(
            model_name=self.model_config['type'],
            input_shape=x_test.shape,
            num_classes=num_classes
        )
        
        # Salvar o modelo inicial
        initial_model_path = os.path.join(self.base_dir, 'model_initial.h5')
        self.model.save(initial_model_path)
        self.logger.info(f"Modelo inicial salvo em {initial_model_path}")

        # Calcular métricas do modelo
        self.calculate_model_metrics()

    def calculate_model_metrics(self):
        """Calcula e armazena métricas sobre o modelo."""
        # Número de parâmetros
        self.metrics['num_parameters'] = self.model.count_params()
        
        # Tamanho em bytes (aproximado)
        total_size = 0
        for weight in self.model.get_weights():
            total_size += weight.nbytes
        self.metrics['model_size_bytes'] = total_size
        
        self.logger.info(f"Modelo tem {self.metrics['num_parameters']:,} parâmetros")
        self.logger.info(f"Tamanho do modelo: {self.metrics['model_size_bytes']/1024/1024:.2f} MB")

    def select_clients(self, round_num: int, available_clients: List[int]) -> List[int]:
        """
        Seleciona os clientes para participar da rodada atual.
        
        Args:
            round_num: Número da rodada atual
            available_clients: Lista de IDs de clientes disponíveis
            
        Returns:
            Lista de IDs de clientes selecionados
        """
        num_clients = self.clients_config['num_clients']
        selection_strategy = self.server_config['selection_strategy']
        selection_fraction = self.server_config['selection_fraction']
        
        num_selected = max(1, int(selection_fraction * len(available_clients)))
        
        if selection_strategy == 'random':
            # Seleção aleatória de clientes
            np.random.seed(self.seed + round_num)  # Seed diferente por rodada
            selected_clients = np.random.choice(
                available_clients, 
                size=num_selected, 
                replace=False
            ).tolist()
        elif selection_strategy == 'all':
            # Seleciona todos os clientes disponíveis
            selected_clients = available_clients
        else:
            # Estratégia desconhecida, usa aleatória
            self.logger.warning(f"Estratégia de seleção '{selection_strategy}' não reconhecida, usando 'random'")
            np.random.seed(self.seed + round_num)
            selected_clients = np.random.choice(
                available_clients, 
                size=num_selected, 
                replace=False
            ).tolist()
        
        self.logger.info(f"Clientes selecionados para a rodada {round_num}: {selected_clients}")
        return selected_clients
    
    def aggregate_models(self, updates: List[Tuple[List[np.ndarray], int]]) -> None:
        """
        Agrega os modelos dos clientes usando a estratégia configurada.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
        """

        # Medir memória antes da agregação
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Medir tempo de agregação
        start_time = time.time()
        
        # try:
        # Obter os client_ids na ordem das atualizações
        client_ids = list(self.round_updates.keys())
        
        # Verificar se a estratégia suporta client_ids
        aggregate_signature = inspect.signature(self.aggregation_strategy.aggregate)
        supports_client_ids = 'client_ids' in aggregate_signature.parameters
        
        if supports_client_ids:
            # Usar a estratégia de agregação configurada com client_ids
            self.logger.info(f"Usando agregação {self.aggregation_strategy.__class__.__name__} com client_ids: {client_ids}")
            aggregated_weights = self.aggregation_strategy.aggregate(updates, client_ids=client_ids)
        else:
            # Fallback para estratégias que não suportam client_ids
            self.logger.info(f"Usando agregação {self.aggregation_strategy.__class__.__name__}")
            aggregated_weights = self.aggregation_strategy.aggregate(updates)
        
        # Aplicar os novos pesos ao modelo global
        self.model.set_weights(aggregated_weights)
        
        if self.save_intermediate_server_models:
            # Salvar o modelo global atualizado
            global_model_path = f'{self.intermediate_server_models_dir}/model_global_round_{self.current_round}.h5'
            self.model.save(global_model_path)
            self.logger.info(f"Modelo global atualizado e salvo em {global_model_path}")
            
        # except Exception as e:
        #     self.logger.error(f"Erro durante a agregação: {str(e)}")
        #     self.logger.warning("Utilizando FedAvg como estratégia de fallback")
            
        #     # Fallback para FedAvg em caso de erro
        #     from src.strategies.aggregation_strategies import FedAvgStrategy
        #     fedavg_strategy = FedAvgStrategy()
        #     aggregated_weights = fedavg_strategy.aggregate(updates)
        #     self.model.set_weights(aggregated_weights)

        # Medir tempo e memória após agregação
        aggregation_time = time.time() - start_time
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_peak = memory_after - memory_before
        
        # Armazenar métricas
        self.metrics['aggregation_time'].append(aggregation_time)
        self.metrics['memory_usage'].append(memory_peak)

        self.logger.info(f"Tempo de agregação: {aggregation_time:.3f}s")
        self.logger.info(f"Pico de memória durante agregação: {memory_peak:.2f} MB")
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Avalia o modelo global nos dados de teste.
        
        Returns:
            Dicionário com métricas de avaliação
        """
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1])
        }
        
        self.logger.info(f"Avaliação do modelo global: {metrics}")
        return metrics
    
    def save_metrics(self) -> None:
        """Salva as métricas do treinamento em um arquivo JSON."""
        metrics_path = os.path.join(self.base_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Métricas salvas em {metrics_path}")
    
    def serialize_model_weights(self) -> List[str]:
        """
        Serializa os pesos do modelo para transferência via API.
        
        Returns:
            Lista de strings base64 representando os pesos do modelo
        """
        weights = self.model.get_weights()
        serialized_weights = []
        
        for w in weights:
            # Serializar cada array numpy para bytes
            bytes_io = io.BytesIO()
            np.save(bytes_io, w, allow_pickle=True)
            bytes_io.seek(0)
            
            # Converter para base64
            serialized = base64.b64encode(bytes_io.read()).decode('utf-8')
            serialized_weights.append(serialized)
        
        return serialized_weights
    
    def deserialize_model_weights(self, serialized_weights: List[str]) -> List[np.ndarray]:
        """
        Desserializa os pesos do modelo recebidos via API.
        
        Args:
            serialized_weights: Lista de strings base64 representando os pesos
            
        Returns:
            Lista de arrays numpy com os pesos do modelo
        """
        weights = []
        
        for w_str in serialized_weights:
            # Decodificar base64 para bytes
            bytes_data = base64.b64decode(w_str.encode('utf-8'))
            bytes_io = io.BytesIO(bytes_data)
            
            # Carregar array numpy
            w = np.load(bytes_io, allow_pickle=True)
            weights.append(w)
        
        return weights

    def start_round(self):
        """
        Inicia uma nova rodada de treinamento.
        
        Returns:
            Dicionário com informações da rodada
        """
        self.current_round += 1
        self.round_start_time = time.time()
        self.logger.info(f"Iniciando rodada {self.current_round}/{self.total_rounds}")
        
        # Limpar atualizações da rodada anterior
        self.round_updates = {}
        
        # Selecionar clientes para esta rodada
        available_clients = list(range(1, self.clients_config['num_clients'] + 1))
        all_selected_clients = self.select_clients(self.current_round, available_clients)
        
        # Inicializar a fila de clientes e clientes ativos
        self.client_queue = all_selected_clients.copy()
        self.active_clients = []
        
        # Ativar o primeiro lote de clientes
        active_count = min(self.max_concurrent_clients, len(self.client_queue))
        self.active_clients = self.client_queue[:active_count]
        self.client_queue = self.client_queue[active_count:]
        
        self.selected_clients = all_selected_clients  # Mantém todos os selecionados
        
        return {
            'round': self.current_round,
            'total_rounds': self.total_rounds,
            'selected_clients': self.selected_clients,
            'active_clients': self.active_clients
        }
    
    def submit_update(
        self, 
        client_id: int, 
        weights: List[str], 
        num_examples: int, 
        local_loss: float, 
        local_accuracy: float
    ) -> Dict[str, Any]:
        """
        Recebe a atualização de um cliente.
        
        Args:
            client_id: ID do cliente
            weights: Pesos do modelo serializados
            num_examples: Número de exemplos usados no treinamento
            
        Returns:
            Dicionário com status da submissão
        """
        # Registrar tempo de início se for primeira submissão
        if client_id not in self.client_start_times:
            self.client_start_times[client_id] = time.time()
            
        if client_id not in self.selected_clients:
            self.logger.warning(f"Cliente {client_id} não foi selecionado para esta rodada.")
            return {'status': 'rejected', 'reason': 'Client not selected for this round'}
        
        if client_id in self.round_updates:
            self.logger.warning(f"Cliente {client_id} já enviou uma atualização para esta rodada.")
            return {'status': 'rejected', 'reason': 'Update already submitted for this round'}

        # Tempo de submissão do cliente
        submission_time = time.time() - self.client_start_times.get(client_id, time.time())

        if client_id not in self.metrics['client_submission_times']:
            self.metrics['client_submission_times'][client_id] = []
        self.metrics['client_submission_times'][client_id].append(submission_time)
        
        # Desserializar pesos
        model_weights = self.deserialize_model_weights(weights)
        
        # Armazenar atualização
        self.round_updates[client_id] = (model_weights, num_examples)

        # Remover dos clientes ativos
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
            
            # Ativar próximo cliente se disponível
            if self.client_queue:
                next_client = self.client_queue.pop(0)
                self.active_clients.append(next_client)
                self.logger.info(f"Ativando próximo cliente: {next_client}")

        if client_id not in self.metrics['local_losses'] and client_id not in self.metrics['local_accuracies'] and client_id not in self.metrics['client_examples']:
            self.metrics['local_losses'][client_id] = []
            self.metrics['local_accuracies'][client_id] = []
            self.metrics['client_examples'][client_id] = []

        self.metrics['local_losses'][client_id].append(local_loss)
        self.metrics['local_accuracies'][client_id].append(local_accuracy)
        self.metrics['client_examples'][client_id].append(num_examples) 

        self.logger.info(f"Atualização recebida do cliente {client_id} com {num_examples} exemplos")
        
        self.check_round_completion()

        return {'status': 'accepted'}

    def check_round_completion(self) -> Dict[str, Any]:
        """
        Verifica se todas as atualizações foram recebidas e, se sim, agrega os modelos.
        """

        # Verificar se todos os clientes selecionados enviaram atualizações
        if set(self.round_updates.keys()) == set(self.selected_clients):
            self.logger.info(f"Todas as atualizações recebidas para a rodada {self.current_round}")
            
            # Agregar modelos
            updates = [(self.round_updates[client_id][0], self.round_updates[client_id][1]) 
                       for client_id in self.selected_clients]
            self.aggregate_models(updates)

            # Limpar a memória após a agregação
            gc.collect()

            # Calcular tempo total da rodada
            round_time = time.time() - self.round_start_time
            self.metrics['round_time'].append(round_time)
            
            # Calcular comunicação total da rodada
            # (2x porque enviamos e recebemos modelos)
            comm_bytes = 2 * self.metrics['model_size_bytes'] * len(self.selected_clients)
            self.metrics['communication_bytes'].append(comm_bytes)
            
            self.logger.info(f"Tempo total da rodada: {round_time:.2f}s")
            self.logger.info(f"Comunicação total: {comm_bytes/1024/1024:.2f} MB")
            
            # Limpar timers para próxima rodada
            self.client_start_times.clear()
            
            # Avaliar modelo se necessário
            if self.current_round % self.server_config['evaluation_interval'] == 0:
                metrics = self.evaluate_model()
                
                # Armazenar métricas
                self.metrics['rounds'].append(self.current_round)
                self.metrics['accuracy'].append(metrics['accuracy'])
                self.metrics['loss'].append(metrics['loss'])
                self.metrics['selected_clients'].append(self.selected_clients)
                self.metrics['num_examples'].append(sum(self.round_updates[client_id][1] for client_id in self.selected_clients))

                # Salvar métricas atualizadas
                self.save_metrics()
            
            # Verificar se é a última rodada
            if self.current_round == self.total_rounds:

                self.training_complete = True
                # Salvar modelo final
                final_model_path = os.path.join(self.base_dir, 'model_final.h5')
                self.model.save(final_model_path)
                self.logger.info(f"Treinamento federado concluído. Modelo final salvo em {final_model_path}")
        return 
    
    def get_round_status(self) -> Dict[str, Any]:
        """
        Retorna o status da rodada atual.
        
        Returns:
            Dicionário com informações do status da rodada
        """
        return {
            'round': self.current_round,
            'total_rounds': self.total_rounds,
            'selected_clients': self.selected_clients,
            'updates_received': list(self.round_updates.keys()),
            'is_round_complete': set(self.round_updates.keys()) == set(self.selected_clients),
            'training_complete': self.training_complete
        }
