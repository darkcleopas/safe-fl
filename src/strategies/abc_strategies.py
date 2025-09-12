from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import logging

class BaseStrategy(ABC):
    """Classe base para todas as estratégias, contendo configuração e logger."""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("FLServer")

class AggregationStrategy(BaseStrategy):
    """Classe abstrata para estratégias de AGREGAÇÃO."""
    
    @abstractmethod
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Agrega os pesos do modelo de vários clientes."""
        pass
    
    def validate_updates(self, updates: List[Tuple[List[np.ndarray], int]]):
        if not updates:
            raise ValueError("Nenhuma atualização recebida para agregação")
        num_layers = len(updates[0][0])
        for weights, _ in updates[1:]:
            if len(weights) != num_layers:
                raise ValueError("As atualizações têm números diferentes de camadas")

class ClientFilterStrategy(BaseStrategy):
    """Classe abstrata para estratégias de FILTRAGEM de clientes."""
    
    @abstractmethod
    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int],
        server_context: Dict[str, Any]
    ) -> List[Tuple[List[np.ndarray], int]]:
        """Filtra as atualizações dos clientes, retornando apenas as aprovadas."""
        pass

class GlobalModelFilterStrategy(BaseStrategy):
    """Classe abstrata para FILTRAGEM da atualização do modelo global."""
    
    @abstractmethod
    def should_update(
        self, 
        new_global_weights: List[np.ndarray],
        old_global_weights: List[np.ndarray],
        server_context: Dict[str, Any]
    ) -> bool:
        """Decide se o modelo global do servidor deve ser substituído pelo novo."""
        pass