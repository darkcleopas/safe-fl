import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
import logging
from collections import defaultdict, deque
from scipy.spatial.distance import cosine


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
        self.logger = logging.getLogger("FLServer")
    
    @abstractmethod
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo de vários clientes.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes correspondentes aos updates
            
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
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando média ponderada pelo número de exemplos.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes (não usado nesta estratégia)
            
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
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando a mediana elementwise.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes (não usado nesta estratégia)
            
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
        self.trim_ratio = self.config.get('trim_ratio', 0.4)
        
        # Validar o trim_ratio
        if self.trim_ratio < 0 or self.trim_ratio >= 0.5:
            self.logger.warning("trim_ratio deve estar entre 0 e 0.5, usando valor padrão de 0.4")
            self.trim_ratio = 0.4

    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando média aparada (trimmed mean).
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes (não usado nesta estratégia)
            
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
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando Krum ou Multi-Krum.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes (usado para logging)
            
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
        
        # Log dos scores com client_ids se disponível
        if client_ids:
            score_info = [f"Cliente {client_ids[i]}: {scores[i]:.4f}" for i in range(len(scores))]
            self.logger.info(f"Scores Krum calculados: {score_info}")
        else:
            self.logger.info(f"Scores Krum calculados: {scores}")
        
        if self.multi_krum:
            # Multi-Krum: selecionar os m modelos com menor score
            if self.num_to_select:
                m = min(self.num_to_select, num_models)
            else:
                m = max(1, num_models - num_malicious)
            
            # Ordenar índices pelo score (crescente) e pegar os m primeiros
            selected_indices = np.argsort(scores)[:m].tolist()
            
            if client_ids:
                selected_clients = [client_ids[i] for i in selected_indices]
                self.logger.info(f"Modelos selecionados pelo Multi-Krum: {selected_clients}")
            else:
                self.logger.info(f"Modelos selecionados pelo Multi-Krum (índices): {selected_indices}")
            
            # Fazer média ponderada dos modelos selecionados
            selected_updates = [(all_weights[i], client_examples[i]) for i in selected_indices]
            return FedAvgStrategy().aggregate(selected_updates)
        else:
            # Krum: selecionar o modelo com menor score
            best_model_idx = int(np.argmin(scores))
            
            if client_ids:
                self.logger.info(f"Modelo selecionado pelo Krum: Cliente {client_ids[best_model_idx]}, score: {scores[best_model_idx]:.4f}")
            else:
                self.logger.info(f"Modelo selecionado pelo Krum: índice {best_model_idx}, score: {scores[best_model_idx]:.4f}")
            
            return all_weights[best_model_idx]


class ClusteringStrategy(AggregationStrategy):
    """
    Implementação da estratégia de agregação baseada em Clustering.
    Separa os modelos em grupos usando agglomerative clustering com base na distância de cosseno.
    """
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Agrega os pesos do modelo usando Clustering.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista opcional de IDs dos clientes (usado para logging)
            
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
        
        if client_ids:
            selected_clients = [client_ids[i] for i in biggest_cluster]
            self.logger.info(f"Selecionados clientes {selected_clients} para agregação de {len(all_weights)} disponíveis")
        else:
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


class CosineSimilarityStrategy(AggregationStrategy):
    """
    Estratégia de agregação baseada em similaridade coseno para detecção de ataques bizantinos.
    
    Detecta anomalias comparando a similaridade coseno entre pesos atuais e anteriores
    de cada cliente. Clientes com baixa similaridade são considerados suspeitos.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a estratégia de similaridade coseno.
        
        Args:
            config: Dicionário com configurações
                threshold: Limiar de similaridade para detectar anomalias (padrão: 0.5)
                min_rounds: Número mínimo de rounds para começar análise (padrão: 2)
                fallback_strategy: Estratégia para usar após filtragem (padrão: 'FED_AVG')
        """
        super().__init__(config)
        
        # Configurações simplificadas
        self.threshold = self.config.get('threshold', 0.5)  # Similaridade mínima esperada
        self.min_rounds = self.config.get('min_rounds', 2)
        self.fallback_strategy = self.config.get('fallback_strategy', 'FED_AVG')
        self.max_clients = self.config.get('max_clients', 1000)  # Limite de clientes no histórico
        
        # Histórico apenas dos pesos anteriores por cliente
        self.previous_weights = {}  # {client_id: last_weights}
        
        # Contador de rounds
        self.round_count = 0
        
        self.logger.info(f"CosineSimilarityStrategy inicializada com threshold={self.threshold}")
    
    def aggregate(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: List[int] = None) -> List[np.ndarray]:
        """
        Agrega os pesos usando detecção de anomalias por similaridade coseno.
        
        Args:
            updates: Lista de tuplas (pesos_modelo, num_exemplos)
            client_ids: Lista de IDs dos clientes correspondente aos updates
            
        Returns:
            Lista de arrays numpy com os pesos agregados
        """
        self.validate_updates(updates)
        self.round_count += 1
        
        # Se client_ids não fornecidos, usar índices sequenciais
        if client_ids is None:
            client_ids = list(range(len(updates)))
            self.logger.warning("client_ids não fornecidos, usando índices sequenciais")
        
        if len(client_ids) != len(updates):
            raise ValueError(f"Número de client_ids ({len(client_ids)}) não corresponde ao número de updates ({len(updates)})")
        
        self.logger.info(f"Round {self.round_count}: analisando {len(updates)} clientes")
        
        # Se temos poucos rounds, usar estratégia padrão sem análise
        if self.round_count < self.min_rounds:
            self.logger.info(f"Round {self.round_count} < {self.min_rounds}, usando FedAvg sem análise")
            self._update_previous_weights(updates, client_ids)
            return FedAvgStrategy().aggregate(updates)
        
        # Filtrar atualizações suspeitas baseado em similaridade coseno
        clean_updates = self._filter_by_cosine_similarity(updates, client_ids)
        
        # Se nenhuma atualização passou no filtro, usar todas (segurança)
        if not clean_updates:
            self.logger.info("Nenhuma atualização passou no filtro, usando todas as atualizações")
            clean_updates = updates
        
        # Atualizar histórico de pesos
        self._update_previous_weights(updates, client_ids)
        
        # Agregar usando estratégia de fallback
        self.logger.info(f"Agregando {len(clean_updates)} de {len(updates)} atualizações")
        return FedAvgStrategy().aggregate(clean_updates)
    
    def _filter_by_cosine_similarity(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: List[int]) -> List[Tuple[List[np.ndarray], int]]:
        """
        Filtra atualizações baseado na similaridade coseno com pesos anteriores.
        
        Args:
            updates: Lista de atualizações
            client_ids: Lista de IDs dos clientes correspondentes
            
        Returns:
            Lista filtrada de atualizações
        """
        clean_updates = []
        
        for i, (weights, num_examples) in enumerate(updates):
            client_id = client_ids[i]
            
            # Verificar se temos pesos anteriores para este cliente
            if client_id not in self.previous_weights:
                # Primeiro round para este cliente - aceitar
                clean_updates.append((weights, num_examples))
                self.logger.info(f"Cliente {client_id}: ACEITO (primeira atualização)")
                continue
            
            # Calcular similaridade coseno
            similarity = self._calculate_cosine_similarity(weights, self.previous_weights[client_id])
            
            # Verificar se está acima do threshold
            if similarity >= self.threshold:
                clean_updates.append((weights, num_examples))
                self.logger.info(f"Cliente {client_id}: ACEITO (similaridade={similarity:.3f})")
            else:
                self.logger.info(f"Cliente {client_id}: REJEITADO (similaridade={similarity:.3f} < {self.threshold})")
        
        return clean_updates
    
    def _calculate_cosine_similarity(self, weights_current: List[np.ndarray], weights_previous: List[np.ndarray]) -> float:
        """
        Calcula a similaridade coseno entre pesos atuais e anteriores.
        
        Args:
            weights_current: Pesos atuais do modelo
            weights_previous: Pesos anteriores do modelo
            
        Returns:
            Valor de similaridade coseno entre 0 e 1 (1 = idênticos, 0 = ortogonais)
        """
        try:
            # Achatar todos os pesos em vetores únicos
            current_flat = np.concatenate([w.flatten() for w in weights_current])
            previous_flat = np.concatenate([w.flatten() for w in weights_previous])
            
            # Verificar se os vetores têm o mesmo tamanho
            if len(current_flat) != len(previous_flat):
                self.logger.error("Tamanhos de vetores diferentes - usando similaridade = 0")
                return 0.0
            
            # Verificar se algum vetor é zero (evita divisão por zero)
            if np.allclose(current_flat, 0) or np.allclose(previous_flat, 0):
                self.logger.warning("Um dos vetores é zero - usando similaridade = 0.0")
                return 0.0
            
            # Calcular similaridade coseno usando scipy
            # cosine() retorna distância (1 - similaridade), então convertemos
            cosine_distance = cosine(current_flat, previous_flat)
            
            # Verificar se o resultado é válido (pode ser NaN se vetores são zeros)
            if np.isnan(cosine_distance):
                self.logger.warning("Similaridade coseno resultou em NaN - usando 0.0")
                return 0.0
            
            # Converter distância para similaridade
            similarity = 1.0 - cosine_distance
            
            # Garantir que está no range [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular similaridade coseno: {e}")
            return 0.0  # Em caso de erro, considerar dissimilar
    
    def _update_previous_weights(self, updates: List[Tuple[List[np.ndarray], int]], client_ids: List[int]):
        """
        Atualiza o histórico de pesos anteriores para todos os clientes.
        
        Args:
            updates: Lista de atualizações
            client_ids: Lista de IDs dos clientes correspondentes
        """
        for i, (weights, _) in enumerate(updates):
            client_id = client_ids[i]
            
            # Armazenar cópia dos pesos atuais como "anteriores" para próximo round
            self.previous_weights[client_id] = [w.copy() for w in weights]
        
        # Limitar número de clientes no histórico para evitar crescimento infinito
        if len(self.previous_weights) > self.max_clients:
            # Remove clientes mais antigos (estratégia FIFO simples)
            # Em implementação real, poderia usar timestamp ou LRU
            excess_clients = len(self.previous_weights) - self.max_clients
            oldest_clients = list(self.previous_weights.keys())[:excess_clients]
            
            for client_id in oldest_clients:
                del self.previous_weights[client_id]
            
            self.logger.info(f"Removidos {excess_clients} clientes antigos do histórico")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da estratégia para análise.
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            'strategy': 'CosineSimilarity',
            'round_count': self.round_count,
            'threshold': self.threshold,
            'clients_tracked': len(self.previous_weights),
            'min_rounds': self.min_rounds,
            'max_clients': self.max_clients,
            'fallback_strategy': self.fallback_strategy
        }
        
        return stats