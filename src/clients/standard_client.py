import logging
import numpy as np
import tensorflow as tf
import requests
import time
import base64
import io
import json
from typing import Dict, Any, Tuple, List, Optional
import gc
import os

from src.utils.model_factory import ModelFactory
from src.utils.dataset_factory import DatasetFactory


class FLClient:
    """
    Cliente base para Federated Learning.
    """
    def __init__(self, client_id: int, config: Dict[str, Any], server_url: str = None):
        """
        Inicializa o cliente com a configuração fornecida.
        
        Args:
            client_id: ID único do cliente
            config: Dicionário com a configuração do cliente
            server_url: URL do servidor FL
        """
        self.client_id = client_id
        self.config = config
        
        # Configurações de subseções
        self.experiment_config = config['experiment']
        self.model_config = config['model']
        self.dataset_config = config['dataset']
        self.clients_config = config['clients']
        self.server_config = config['server']
        
        # URL do servidor
        if server_url:
            self.server_url = server_url
        else:
            self.server_url = f"http://{self.server_config['address']}"
        
        # Definir seed para reprodutibilidade
        self.seed = self.experiment_config.get('seed', 42)
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        
        # Configurar logging
        self.setup_logging()
        
        # Configurar recurso computacional
        self.setup_resource()
        
        # Carregar dados
        self.load_data()
        
        # O modelo será recebido do servidor
        self.model = None
        
        # Estado do treinamento
        self.current_round = 0
        self.is_selected = False
        
        self.logger.info(f"Cliente {client_id} inicializado com sucesso")
    
    def setup_logging(self):
        """Configura o sistema de logging."""
        log_level = self.experiment_config.get('log_level', 'info').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"FLClient-{self.client_id}")
    
    def setup_resource(self):
        """Define os recursos computacionais do cliente."""        
        # Limitar número de threads
        tf_num_threads = int(os.environ.get('TF_NUM_THREADS', '1'))
        
        # Limitar threads
        tf.config.threading.set_intra_op_parallelism_threads(tf_num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)
        
        self.logger.info(f"TensorFlow configurado com {tf_num_threads} threads")
    
    def load_data(self):
        """Carrega os dados do cliente."""
        
        dataset_factory = DatasetFactory()
        self.x_train, self.y_train, self.x_test, self.y_test, self.num_classes = dataset_factory.load_dataset(
            dataset_name=self.dataset_config['name'],
            client_id=self.client_id,
            num_clients=self.clients_config['num_clients'],
            non_iid=self.dataset_config['non_iid'],
            seed=self.seed
        )
        
        self.num_examples = len(self.x_train)
        self.logger.info(f"Dados carregados: {self.num_examples} exemplos de treinamento")
    
    def serialize_model_weights(self, weights: List[np.ndarray]) -> List[str]:
        """
        Serializa os pesos do modelo para transferência via API.
        
        Args:
            weights: Lista de arrays numpy com os pesos do modelo
            
        Returns:
            Lista de strings base64 representando os pesos
        """
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
    
    def fetch_model(self) -> bool:
        """
        Busca o modelo global do servidor.
        
        Returns:
            True se o modelo foi atualizado com sucesso, False caso contrário
        """
        try:
            self.logger.info("Buscando modelo global do servidor")
            response = requests.get(f"{self.server_url}/model")
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao buscar modelo: {response.status_code}")
                return False
            
            data = response.json()
            weights = self.deserialize_model_weights(data['weights'])
            round_num = data['round']
            
            # Atualizar o modelo com os novos pesos
            if self.model is None:
                self.initialize_model(weights)
            else:
                self.model.set_weights(weights)
            
            self.current_round = round_num
            self.logger.info(f"Modelo atualizado com sucesso (Rodada {round_num})")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar modelo: {str(e)}")
            return False
    
    def initialize_model(self, weights: List[np.ndarray]):
        """
        Inicializa o modelo com os pesos fornecidos.
        
        Args:
            weights: Lista de arrays numpy com os pesos do modelo
        """
        
        # Inicializar o modelo
        model_factory = ModelFactory()
        self.model = model_factory.create_model(
            model_name=self.model_config['type'],
            input_shape=self.x_train.shape,
            num_classes=self.num_classes
        )
        
        # Definir os pesos
        self.model.set_weights(weights)
        self.logger.info("Modelo inicializado com os pesos do servidor")
    
    def check_round(self) -> Tuple[bool, bool]:
        """
        Verifica o status da rodada atual no servidor.
        
        Returns:
            Tupla (is_selected, training_complete)
        """
        try:
            # Verificar status geral da rodada
            response = requests.get(f"{self.server_url}/round")
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao verificar rodada: {response.status_code}")
                return False, False, False, False
            
            data = response.json()
            self.current_round = data['round']
            training_complete = data['training_complete']
            
            # Verificar status específico deste cliente
            client_response = requests.get(f"{self.server_url}/client_status/{self.client_id}")
            
            if client_response.status_code != 200:
                self.logger.error(f"Erro ao verificar status do cliente: {client_response.status_code}")
                return False, False, False, False
            
            client_data = client_response.json()
            
            is_selected = client_data['is_selected']
            is_active = client_data['is_active']
            is_completed = client_data['is_completed']
            
            if is_selected and is_active and not is_completed:
                self.logger.info(f"Cliente ativo para a rodada {self.current_round}")
            elif is_selected and not is_active and not is_completed:
                self.logger.info(f"Cliente selecionado mas aguardando ativação para a rodada {self.current_round}")
            
            return is_selected, training_complete, is_completed, is_active
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar rodada: {str(e)}")
            return False, False, False, False
    
    def train_model(self) -> Tuple[List[np.ndarray], int]:
        """
        Treina o modelo local com os dados do cliente.
        
        Returns:
            Tupla contendo os pesos do modelo treinado e o número de exemplos usados
        """
        if self.model is None:
            self.logger.error("Modelo não inicializado. É necessário buscar o modelo primeiro.")
            raise ValueError("Modelo não inicializado")
        
        # Parâmetros de treinamento
        local_epochs = self.model_config['local_epochs']
        batch_size = self.model_config['batch_size']
        
        # Ajustar o número de épocas com base na capacidade computacional
        # Clientes com menor capacidade podem treinar menos épocas
        # effective_epochs = max(1, int(local_epochs * (self.computation_capability / 4.0)))
        effective_epochs = local_epochs
        self.logger.info(f"Iniciando treinamento local por {effective_epochs} épocas")
        
        # Treinar o modelo
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=effective_epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Logs de métricas
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        self.logger.info(f"Treinamento concluído. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        # Guardar os pesos antes de limpar a sessão
        weights = self.model.get_weights().copy()
        
        # Rodar coleta de lixo para liberar memória
        tf.keras.backend.clear_session()
        gc.collect()
        
        return weights, self.num_examples, final_loss, final_accuracy
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Avalia o modelo local nos dados de teste do cliente.
        
        Returns:
            Dicionário com métricas de avaliação
        """
        if self.model is None:
            self.logger.error("Modelo não inicializado. É necessário buscar o modelo primeiro.")
            raise ValueError("Modelo não inicializado")
        
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1])
        }
        
        self.logger.info(f"Avaliação do modelo local: {metrics}")
        return metrics
    
    def submit_update(
        self,
        weights: List[np.ndarray],
        num_examples: int,
        loss: float,
        accuracy: float
    ) -> bool:
        """
        Envia os pesos do modelo treinado para o servidor.
        
        Args:
            weights: Pesos do modelo treinado
            num_examples: Número de exemplos usados no treinamento
            
        Returns:
            True se o envio foi bem-sucedido, False caso contrário
        """
        try:
            # Serializar os pesos
            serialized_weights = self.serialize_model_weights(weights)
            
            # Preparar payload
            payload = {
                'client_id': self.client_id,
                'weights': serialized_weights,
                'num_examples': num_examples,
                'local_loss': loss,
                'local_accuracy': accuracy
            }
            
            # Enviar para o servidor
            self.logger.info(f"Enviando atualização para o servidor (Rodada {self.current_round})")
            response = requests.post(f"{self.server_url}/update", json=payload)
            
            if response.status_code != 200:
                self.logger.error(f"Erro ao enviar atualização: {response.status_code}")
                return False
            
            data = response.json()
            if data.get('status') == 'accepted':
                self.logger.info("Atualização aceita pelo servidor")
                return True
            else:
                self.logger.warning(f"Atualização rejeitada: {data.get('reason')}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar atualização: {str(e)}")
            return False
    
    def training_loop(self):
        """
        Loop principal de treinamento do cliente.
        """
        self.logger.info("Iniciando loop de treinamento")
        
        try:
            # Buscar modelo inicial
            if not self.fetch_model():
                self.logger.error("Não foi possível obter o modelo inicial")
                return
            
            while True:
                # Verificar status da rodada e se o cliente está ativo
                is_selected, training_complete, is_round_complete_for_client, is_active = self.check_round()
                
                if training_complete:
                    self.logger.info("Treinamento federado concluído")
                    self.fetch_model()
                    metrics = self.evaluate_model()
                    self.logger.info(f"Métricas finais: {json.dumps(metrics)}")
                    tf.keras.backend.clear_session()
                    gc.collect()
                    break

                if is_round_complete_for_client:
                    self.logger.info("Atualização já enviada para a rodada atual. Aguardando próxima rodada...")
                    time.sleep(10)
                    continue
                
                # Se selecionado mas não ativo, aguardar na fila
                if is_selected and not is_active:
                    self.logger.info("Cliente aguardando na fila. Verificando novamente em breve...")
                    time.sleep(5)  # Verificar com mais frequência quando está na fila
                    continue
                
                if is_selected and is_active:
                    # Buscar o modelo atual
                    if not self.fetch_model():
                        self.logger.error("Falha ao buscar o modelo. Tentando novamente...")
                        time.sleep(5)
                        continue
                    
                    # Treinar o modelo
                    weights, num_examples, loss, accuracy = self.train_model()
                    
                    # Enviar atualização
                    if not self.submit_update(weights, num_examples, loss, accuracy):
                        self.logger.error("Falha ao enviar atualização. Tentando novamente...")
                        time.sleep(5)
                        continue
                    
                    # Limpar recursos
                    tf.keras.backend.clear_session()
                    gc.collect()

                # Aguardar antes da próxima verificação
                time.sleep(10)
                
        except KeyboardInterrupt:
            self.logger.info("Treinamento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro no loop de treinamento: {str(e)}")
        
        # Limpar recursos
        tf.keras.backend.clear_session()
        gc.collect()
