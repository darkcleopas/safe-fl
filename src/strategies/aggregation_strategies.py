import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats

from src.strategies.abc_strategies import AggregationStrategy

class FedAvgStrategy(AggregationStrategy):
    """Implementação da estratégia de agregação FedAvg (média ponderada)."""
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        self.validate_updates(updates)
        self.logger.info("Aplicando agregação FedAvg")
        
        total_examples = sum(num_examples for _, num_examples in updates)
        if total_examples == 0:
            # Retorna o primeiro modelo como fallback se não houver exemplos
            return updates[0][0]
        
        weighted_weights = [
            [layer * num_examples for layer in weights] 
            for weights, num_examples in updates
        ]
        
        avg_weights = [
            np.sum([weights[i] for weights in weighted_weights], axis=0) / total_examples
            for i in range(len(weighted_weights[0]))
        ]
        
        return avg_weights


class FedMedianStrategy(AggregationStrategy):
    """Implementação da estratégia de agregação FedMedian."""
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        self.validate_updates(updates)
        self.logger.info("Aplicando agregação FedMedian")
        
        all_weights = [weights for weights, _ in updates]
        aggregated_weights = []
        
        for layer_idx in range(len(all_weights[0])):
            layer_weights = [model_weights[layer_idx] for model_weights in all_weights]
            stacked_weights = np.stack(layer_weights, axis=0)
            median_weights = np.median(stacked_weights, axis=0)
            aggregated_weights.append(median_weights)
        
        return aggregated_weights


class TrimmedMeanStrategy(AggregationStrategy):
    """Implementação da estratégia de agregação Trimmed Mean."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.trim_ratio = self.config.get('trim_ratio', 0.4)
        if self.trim_ratio < 0 or self.trim_ratio >= 0.5:
            self.logger.warning("trim_ratio deve estar entre 0 e 0.5, usando valor padrão de 0.4")
            self.trim_ratio = 0.4

    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        self.validate_updates(updates)
        self.logger.info(f"Aplicando agregação Trimmed Mean com trim_ratio={self.trim_ratio}")
        
        all_weights = [weights for weights, _ in updates]
        num_models = len(all_weights)
        num_to_trim = int(num_models * self.trim_ratio)
        aggregated_weights = []
        
        for layer_idx in range(len(all_weights[0])):
            layer_weights = [model_weights[layer_idx] for model_weights in all_weights]
            stacked_weights = np.stack(layer_weights, axis=0)
            
            original_shape = stacked_weights.shape[1:]
            reshaped_weights = stacked_weights.reshape(num_models, -1)
            trimmed_mean = np.zeros(reshaped_weights.shape[1])
            
            for i in range(reshaped_weights.shape[1]):
                values = reshaped_weights[:, i]
                sorted_values = np.sort(values)
                if num_to_trim > 0:
                    trimmed_values = sorted_values[num_to_trim:-num_to_trim]
                else:
                    trimmed_values = sorted_values
                trimmed_mean[i] = np.mean(trimmed_values)
            
            aggregated_layer = trimmed_mean.reshape(original_shape)
            aggregated_weights.append(aggregated_layer)
        
        return aggregated_weights