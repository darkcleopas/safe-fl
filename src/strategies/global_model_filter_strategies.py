import numpy as np
from typing import List, Dict, Any

from src.strategies.abc_strategies import GlobalModelFilterStrategy

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