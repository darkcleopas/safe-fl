import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import logging


class AggregationStrategy(ABC):
    """
    Classe abstrata que define a interface para estratégias de agregação.
    
    Todas as estratégias de agregação devem implementar este contrato.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a estratégia de agregação com configurações opcionais.
        
        Args:
            config: Dicionário com configurações específicas da estratégia
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo de vários clientes.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        pass
    
    def validate_updates(self, updates: List[Tuple[List[np.ndarray], int]]) -> bool:
        """
        Valida as atualizações recebidas. Método auxiliar para verificar
        se as atualizações são válidas para agregação.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            True se as atualizações são válidas, False caso contrário
        
        Raises:
            ValueError: Se as atualizações forem inválidas
        """
        if not updates:
            raise ValueError("Nenhuma atualização recebida para agregação")
        
        # Verificar se todas as atualizações têm o mesmo número de camadas
        num_layers = len(updates[0][0])
        for weights, _ in updates[1:]:
            if len(weights) != num_layers:
                raise ValueError("As atualizações têm números diferentes de camadas")
        
        return True


class FedAvgStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação FedAvg (média ponderada).
    """
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando média ponderada pelo número de exemplos.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        
        self.logger.info("Aplicando agregação FedAvg")
        
        # Calcular o total de exemplos
        total_examples = sum(num_examples for _, num_examples in updates)
        
        if total_examples == 0:
            raise ValueError("Número total de exemplos é zero. Não é possível agregar os modelos.")
        
        # Obter os pesos de cada modelo e multiplicar pelo número de exemplos
        weighted_weights = [
            [layer * num_examples for layer in weights] 
            for weights, num_examples in updates
        ]
        
        # Calcular a média ponderada para cada camada
        avg_weights = [
            np.sum([weights[i] for weights in weighted_weights], axis=0) / total_examples
            for i in range(len(weighted_weights[0]))
        ]
        
        return avg_weights


class FedMedianStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação FedMedian.
    """
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando a mediana elementwise.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        
        self.logger.info("Aplicando agregação FedMedian")
        
        # Extrair apenas os pesos dos modelos, ignorando o número de exemplos
        all_weights = [weights for weights, _ in updates]
        
        # Inicializar a lista para os pesos agregados
        aggregated_weights = []
        
        # Para cada camada do modelo
        for layer_idx in range(len(all_weights[0])):
            # Coletar os pesos desta camada de todos os modelos
            layer_weights = [model_weights[layer_idx] for model_weights in all_weights]
            
            # Empilhar para calcular a mediana ao longo do eixo 0
            stacked_weights = np.stack(layer_weights, axis=0)
            
            # Calcular a mediana para cada elemento
            median_weights = np.median(stacked_weights, axis=0)
            
            # Adicionar à lista de pesos agregados
            aggregated_weights.append(median_weights)
        
        return aggregated_weights


class TrimmedMeanStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação Trimmed Mean.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a estratégia de agregação Trimmed Mean.
        
        Args:
            config: Dicionário com configurações
                trim_ratio: Porcentagem (0-0.5) de valores extremos a remover
        """
        super().__init__(config)
        self.trim_ratio = self.config.get('trim_ratio', 0.1)
        
        # Validar o trim_ratio
        if self.trim_ratio < 0 or self.trim_ratio >= 0.5:
            self.logger.warning("trim_ratio deve estar entre 0 e 0.5, usando valor padrão de 0.1")
            self.trim_ratio = 0.1
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando média aparada (trimmed mean).
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        
        self.logger.info(f"Aplicando agregação Trimmed Mean com trim_ratio={self.trim_ratio}")
        
        # Extrair apenas os pesos dos modelos, ignorando o número de exemplos
        all_weights = [weights for weights, _ in updates]
        
        # Calcular quantos modelos cortar em cada extremo
        num_models = len(all_weights)
        num_to_trim = int(num_models * self.trim_ratio)
        
        # Inicializar a lista para os pesos agregados
        aggregated_weights = []
        
        # Para cada camada do modelo
        for layer_idx in range(len(all_weights[0])):
            # Coletar os pesos desta camada de todos os modelos
            layer_weights = [model_weights[layer_idx] for model_weights in all_weights]
            
            # Empilhar para ordenação e corte
            stacked_weights = np.stack(layer_weights, axis=0)
            
            # Para cada elemento na camada, ordenar, cortar e calcular a média
            # Reshape para facilitar o processamento elemento a elemento
            original_shape = stacked_weights.shape[1:]
            reshaped_weights = stacked_weights.reshape(num_models, -1)
            
            # Inicializar array para os resultados
            trimmed_mean = np.zeros(reshaped_weights.shape[1])
            
            # Para cada elemento/parâmetro
            for i in range(reshaped_weights.shape[1]):
                # Extrair valores para este elemento de todos os modelos
                values = reshaped_weights[:, i]
                
                # Ordenar, cortar e calcular a média
                sorted_values = np.sort(values)
                if num_to_trim > 0:
                    trimmed_values = sorted_values[num_to_trim:-num_to_trim]
                else:
                    trimmed_values = sorted_values
                    
                trimmed_mean[i] = np.mean(trimmed_values)
            
            # Reshape de volta para a forma original
            aggregated_layer = trimmed_mean.reshape(original_shape)
            aggregated_weights.append(aggregated_layer)
        
        return aggregated_weights


class KrumStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação Krum.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a estratégia de agregação Krum.
        
        Args:
            config: Dicionário com configurações
                num_malicious: Número estimado de clientes maliciosos
                multi_krum: Se True, usa Multi-Krum (seleciona múltiplos modelos)
                malicious_percentage: Porcentagem de clientes maliciosos (usado se num_malicious não for fornecido)
        """
        super().__init__(config)
        self.num_malicious = self.config.get('num_malicious')
        self.multi_krum = self.config.get('multi_krum', False)
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando Krum ou Multi-Krum.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        
        if self.multi_krum:
            self.logger.info("Aplicando agregação Multi-Krum")
        else:
            self.logger.info("Aplicando agregação Krum")
        
        # Extrair apenas os pesos dos modelos e o número de exemplos
        all_weights = [weights for weights, _ in updates]
        client_examples = [num_examples for _, num_examples in updates]
        
        # Verificar se temos atualizações suficientes
        num_models = len(all_weights)
        if num_models < 3:
            self.logger.warning(f"Número insuficiente de modelos para Krum ({num_models}), usando FedAvg")
            return FedAvgStrategy().aggregate(updates)
        
        # Se não for especificado, tentar estimar o número de clientes maliciosos
        num_malicious = self.num_malicious
        if num_malicious is None:
            # Usar a porcentagem configurada
            malicious_percentage = self.config.get('malicious_percentage', 0.2)
            num_malicious = max(1, int(num_models * malicious_percentage))
            self.logger.info(f"Número de clientes maliciosos estimado em {num_malicious}")
        
        # Garantir que num_malicious seja válido
        if num_malicious >= num_models / 2:
            self.logger.warning(f"Número de clientes maliciosos muito alto ({num_malicious}), ajustando")
            num_malicious = num_models // 2 - 1
        
        # Achatar os pesos para calcular distâncias entre modelos
        flattened_weights = []
        for weights in all_weights:
            client_flat = np.concatenate([w.flatten() for w in weights])
            flattened_weights.append(client_flat)
        
        # Calcular distâncias euclidianas entre todos os pares de modelos
        distances = np.zeros((num_models, num_models))
        
        for i in range(num_models):
            for j in range(i+1, num_models):
                dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Para cada modelo, calcular a soma das distâncias para os k modelos mais próximos
        k = num_models - num_malicious - 2  # conforme o algoritmo Krum
        if k < 1:
            k = 1
        
        scores = []
        for i in range(num_models):
            # Ordenar distâncias e pegar as k menores (excluindo a distância a si mesmo, que é 0)
            closest_distances = np.sort(distances[i])[1:k+1]
            scores.append(np.sum(closest_distances))
        
        if self.multi_krum:
            # Multi-Krum: selecionar os m modelos com menor score
            m = num_models - num_malicious
            selected_indices = np.argsort(scores)[:m]
            self.logger.info(f"Modelos selecionados pelo Multi-Krum: {selected_indices}")
            
            # Fazer média ponderada dos modelos selecionados
            selected_updates = [(all_weights[i], client_examples[i]) for i in selected_indices]
            return FedAvgStrategy().aggregate(selected_updates)
        else:
            # Krum: selecionar o modelo com menor score
            best_model_idx = np.argmin(scores)
            self.logger.info(f"Modelo selecionado pelo Krum: cliente com índice {best_model_idx}")
            return all_weights[best_model_idx]
