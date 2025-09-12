import numpy as np
from typing import List, Dict, Any, Deque
from collections import deque

from src.strategies.abc_strategies import GlobalModelFilterStrategy

class AdaptiveL2GlobalModelFilter(GlobalModelFilterStrategy):
    """
    Decide se a atualização do modelo global deve ser aceita com base na
    distância L2 em relação ao modelo anterior, usando um threshold adaptativo
    calculado a partir de uma janela móvel.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.window_size = self.config.get('window_size', 7)
        self.std_dev_multiplier = self.config.get('std_dev_multiplier', 1.5)
        self.min_rounds_history = self.config.get('min_rounds_history', 5)

        # Deque para armazenar o histórico de distâncias das atualizações globais
        self.update_distance_history: Deque[float] = deque(maxlen=self.window_size)

    def should_update(
        self, 
        new_global_weights: List[np.ndarray],
        old_global_weights: List[np.ndarray],
        server_context: Dict[str, Any]
    ) -> bool:
        if server_context.get('previous_global_weights') is None:
            self.logger.info("Filtro L2 Global Adaptativo: Sem modelo anterior, atualização aceita.")
            return True
        
        # Calcula a distância da atualização atual
        new_flat = np.concatenate([w.flatten() for w in new_global_weights])
        old_flat = np.concatenate([w.flatten() for w in old_global_weights])
        distance = np.linalg.norm(new_flat - old_flat)

        current_round = server_context.get('round', 0)
        if current_round < self.min_rounds_history:
            self.logger.info(f"Filtro L2 Global Adaptativo: Round {current_round} < {self.min_rounds_history}, atualização aceita.")
            self.update_distance_history.append(distance)
            return True
        
        median_dist = np.median(self.update_distance_history)
        std_dist = np.std(self.update_distance_history)
        adaptive_threshold = median_dist + self.std_dev_multiplier * std_dist
        
        if distance <= adaptive_threshold:
            self.logger.info(f"Filtro L2 Global: Atualização ACEITA (dist={distance:.3f} <= {adaptive_threshold:.3f})")
            self.update_distance_history.append(distance) # Adiciona ao histórico
            return True
        else:
            self.logger.warning(f"Filtro L2 Global: Atualização REJEITADA (dist={distance:.3f} > {adaptive_threshold:.3f})")
            return False

class L2GlobalModelFilter(GlobalModelFilterStrategy):
    """
    Decide se a atualização do modelo global deve ser aceita com base na
    distância L2 em relação ao modelo anterior.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.update_threshold = self.config.get('update_threshold', 2.0)
        self.min_rounds = self.config.get('min_rounds', 5)

    def should_update(
        self, 
        new_global_weights: List[np.ndarray],
        old_global_weights: List[np.ndarray],
        server_context: Dict[str, Any]
    ) -> bool:
        if server_context.get('previous_global_weights') is None:
            self.logger.info("Filtro L2 Global: Sem modelo anterior, atualização aceita.")
            return True
        
        current_round = server_context.get('round', 0)
        if current_round < self.min_rounds:
            self.logger.info(f"Filtro L2 Global: Round {current_round} < {self.min_rounds}, atualização aceita.")
            return True

        new_flat = np.concatenate([w.flatten() for w in new_global_weights])
        old_flat = np.concatenate([w.flatten() for w in old_global_weights])
        
        distance = np.linalg.norm(new_flat - old_flat)
        
        if distance <= self.update_threshold:
            self.logger.info(f"Filtro L2 Global: Atualização ACEITA (distancia={distance:.3f} <= {self.update_threshold})")
            return True
        else:
            self.logger.warning(f"Filtro L2 Global: Atualização REJEITADA (distancia={distance:.3f} > {self.update_threshold})")
            return False