import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


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
    
    O algoritmo Krum seleciona o modelo com a menor soma de distâncias para seus k modelos mais próximos,
    onde k = n - f - 2 (n é o número total de modelos e f é o número estimado de modelos maliciosos).
    
    Multi-Krum é uma variação que seleciona os m modelos com as menores somas de distâncias e então
    calcula a média ponderada desses modelos.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a estratégia de agregação Krum.
        
        Args:
            config: Dicionário com configurações
                num_malicious: Número estimado de clientes maliciosos
                multi_krum: Se True, usa Multi-Krum (seleciona múltiplos modelos)
                malicious_percentage: Porcentagem de clientes maliciosos (usado se num_malicious não for fornecido)
                num_to_select: Em Multi-Krum, número de modelos a selecionar (padrão: n-f)
        """
        super().__init__(config)
        self.num_malicious = self.config.get('num_malicious')
        self.multi_krum = self.config.get('multi_krum', False)
        # Número de modelos a selecionar no Multi-Krum (se None, usa n-f)
        self.num_to_select = self.config.get('num_to_select')
    
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
            self.logger.warning(f"Número de clientes maliciosos muito alto ({num_malicious}), ajustando para (n/2)-1")
            num_malicious = max(1, num_models // 2 - 1)
        
        # Achatar os pesos para calcular distâncias entre modelos
        flattened_weights = []
        for weights in all_weights:
            client_flat = np.concatenate([w.flatten() for w in weights])
            flattened_weights.append(client_flat)
        
        # Calcular distâncias euclidianas entre todos os pares de modelos
        distances = np.zeros((num_models, num_models))
        
        for i in range(num_models):
            for j in range(num_models):
                if i == j:
                    distances[i, j] = 0  # Distância de um modelo para si mesmo é 0
                else:
                    dist = np.linalg.norm(flattened_weights[i] - flattened_weights[j])
                    distances[i, j] = dist**2  # Usamos distância euclidiana ao quadrado
        
        # Para cada modelo, calcular a soma das distâncias para os k modelos mais próximos
        k = num_models - num_malicious - 2  # conforme o algoritmo Krum
        if k < 1:
            k = 1
            self.logger.warning(f"k foi ajustado para 1 devido ao baixo número de modelos")
        
        scores = []
        for i in range(num_models):
            # Ordenar distâncias e pegar as k menores (excluindo a distância a si mesmo, que é 0)
            closest_distances = np.sort(distances[i])[1:k+1]
            scores.append(np.sum(closest_distances))
        
        self.logger.info(f"Scores Krum calculados: {scores}")
        
        if self.multi_krum:
            # Multi-Krum: selecionar os m modelos com menor score
            if self.num_to_select:
                m = min(self.num_to_select, num_models)
            else:
                m = max(1, num_models - num_malicious)
            
            # Ordenar índices pelo score (crescente) e pegar os m primeiros
            selected_indices = np.argsort(scores)[:m].tolist()
            self.logger.info(f"Modelos selecionados pelo Multi-Krum: {selected_indices}")
            
            # Fazer média ponderada dos modelos selecionados
            selected_updates = [(all_weights[i], client_examples[i]) for i in selected_indices]
            return FedAvgStrategy().aggregate(selected_updates)
        else:
            # Krum: selecionar o modelo com menor score
            best_model_idx = int(np.argmin(scores))
            self.logger.info(f"Modelo selecionado pelo Krum: cliente com índice {best_model_idx}, score: {scores[best_model_idx]}")
            return all_weights[best_model_idx]


class ClusteringStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação baseada em Clustering.
    Separa os modelos em grupos usando agglomerative clustering com base na distância de cosseno.
    """
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando Clustering.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        
        self.logger.info("Aplicando agregação Clustering")
        
        # Extrair apenas os pesos dos modelos e o número de exemplos
        all_weights = [weights for weights, _ in updates]
        num_examples = [n_examples for _, n_examples in updates]
        
        # Caso especial: se houver apenas um modelo, retorná-lo diretamente
        if len(all_weights) == 1:
            self.logger.info("Apenas um modelo disponível, retornando-o diretamente")
            return all_weights[0]
        
        # Caso especial: se houver apenas dois modelos, calcular a média
        if len(all_weights) == 2:
            self.logger.info("Apenas dois modelos disponíveis, calculando a média")
            return FedAvgStrategy().aggregate(updates)
        
        # Identificar o maior cluster
        biggest_cluster = self._clustering_filtering(all_weights)
        
        if not biggest_cluster:
            self.logger.warning("Nenhum cluster identificado, usando FedAvg com todos os modelos")
            return FedAvgStrategy().aggregate(updates)
        
        # Selecionar modelos do maior cluster
        selected_updates = [(all_weights[i], num_examples[i]) for i in biggest_cluster]
        self.logger.info(f"Selecionados {len(biggest_cluster)} modelos para agregação de {len(all_weights)} disponíveis")
        
        # Usar FedAvg para calcular a média dos modelos selecionados
        return FedAvgStrategy().aggregate(selected_updates)
    
    def _clustering_filtering(self, models: List[List[np.ndarray]]) -> List[int]:
        """
        Separa os modelos em dois grupos usando agglomerative clustering com average link
        baseado na matriz de distância de cosseno entre cada par de modelos.
        
        Args:
            models: Lista de modelos (cada modelo é uma lista de arrays numpy)
            
        Returns:
            Lista de índices correspondentes ao maior cluster
        """
        if len(models) <= 2:
            return list(range(len(models)))  # Retornar todos os índices para casos simples
        
        # Compute cosine similarities between vectors
        similarity_matrix = self._compute_cosine_similarity_matrix(models)
        
        # Transform the similarity matrix into a condensed distance matrix
        try:
            biggest_cluster = self._agglomerative_clustering(squareform(similarity_matrix))
            return biggest_cluster
        except Exception as e:
            self.logger.error(f"Erro ao realizar clustering: {str(e)}")
            return list(range(len(models)))  # Em caso de erro, usar todos os modelos
    
    def _agglomerative_clustering(self, agglomerative_distance: np.ndarray) -> List[int]:
        """
        Aplica agglomerative clustering e retorna o maior cluster.
        
        Args:
            agglomerative_distance: Matriz de distância condensada
            
        Returns:
            Lista de índices correspondentes ao maior cluster
        """
        # Apply agglomerative clustering with average link
        Z = linkage(agglomerative_distance, 'average')
        Z = np.abs(Z)
        
        # Generate two clusters
        clusters = fcluster(Z, t=2, criterion='maxclust')
        
        # Extracting the indices of the biggest cluster
        cluster_1 = [i for i, c in enumerate(clusters) if c == 1]
        cluster_2 = [i for i, c in enumerate(clusters) if c == 2]
        
        # Log information about clusters
        self.logger.info(f"Cluster 1 tamanho: {len(cluster_1)}, Cluster 2 tamanho: {len(cluster_2)}")
        
        return cluster_1 if len(cluster_1) >= len(cluster_2) else cluster_2
    
    def _compute_cosine_similarity_matrix(self, models: List[List[np.ndarray]]) -> np.ndarray:
        """
        Computa uma matriz com a distância/similaridade de cosseno entre cada par de modelos.
        
        Args:
            models: Lista de modelos (cada modelo é uma lista de arrays numpy)
            
        Returns:
            Matriz de similaridade de cosseno
        """
        # Flatten model weights for easier computation
        flattened_models = []
        for model in models:
            flattened = np.concatenate([w.flatten() for w in model])
            flattened_models.append(flattened)
        
        num_models = len(flattened_models)
        similarity_matrix = np.zeros((num_models, num_models))
        
        for i in range(num_models):
            for j in range(num_models):
                if i == j:
                    similarity_matrix[i, j] = 0  # Distância de um modelo para si mesmo é 0
                else:
                    similarity = self._compute_cosine_similarity(flattened_models[i], flattened_models[j])
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _compute_cosine_similarity(self, model_1: np.ndarray, model_2: np.ndarray) -> float:
        """
        Computa a similaridade de cosseno entre dois vetores.
        
        Args:
            model_1: Primeiro vetor
            model_2: Segundo vetor
            
        Returns:
            Similaridade de cosseno (valor entre 0 e 1, onde 1 indica vetores idênticos)
        """
        norm_1 = np.linalg.norm(model_1)
        norm_2 = np.linalg.norm(model_2)
        
        if norm_1 == 0 or norm_2 == 0:
            return 0
        
        inner_product = np.dot(model_1, model_2)
        cosine_similarity = 1 - (inner_product / (norm_1 * norm_2))
        
        # Garante que a similaridade esteja no intervalo [0, 1]
        return max(0, min(1, cosine_similarity))