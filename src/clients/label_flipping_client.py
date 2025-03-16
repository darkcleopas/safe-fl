import numpy as np
import logging
from src.clients.standard_client import FLClient


class FLClient(FLClient):
    """
    Cliente malicioso que implementa um ataque de label flipping.
    Herda do cliente padrão e modifica apenas os dados durante o carregamento.
    """
    
    def __init__(self, client_id, config, server_url=None):
        """
        Inicializa o cliente malicioso.
        
        Args:
            client_id: ID único do cliente
            config: Dicionário com a configuração do cliente
            server_url: URL do servidor FL
        """
        super().__init__(client_id, config, server_url)
        self.logger.info(f"Cliente malicioso {client_id} (label_flipping) inicializado")
        
    def load_data(self):
        """
        Carrega e manipula os dados para executar o ataque de label flipping.
        Substitui o método original para corromper os rótulos.
        """
        # Chama o método da classe base para carregar os dados originais
        super().load_data()
        
        # Log original antes da manipulação
        self.logger.info(f"Cliente malicioso {self.client_id}: Dados originais carregados - {self.num_examples} exemplos")
        
        # Implementar o ataque de label flipping
        self.logger.warning(f"Cliente malicioso {self.client_id}: Realizando ataque de label flipping")
        
        # Determinar o número de classes do conjunto de dados
        num_classes = self.y_train.shape[1]
        
        # Inverter uma porcentagem dos rótulos (50%)
        indices_to_flip = np.random.choice(len(self.y_train), size=int(len(self.y_train) * 0.5), replace=False)
        
        # Para cada índice selecionado, trocar o rótulo
        for idx in indices_to_flip:
            # Obter o rótulo atual
            current_label = np.argmax(self.y_train[idx])
            # Escolher um novo rótulo diferente do atual
            new_label = (current_label + 1) % num_classes
            
            # Criar um novo vetor one-hot para o novo rótulo
            new_one_hot = np.zeros(num_classes)
            new_one_hot[new_label] = 1
            
            # Substituir o rótulo
            self.y_train[idx] = new_one_hot
        
        self.logger.warning(f"Cliente malicioso {self.client_id}: {len(indices_to_flip)} rótulos invertidos")