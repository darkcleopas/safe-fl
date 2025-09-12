import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.strategies.abc_strategies import ClientFilterStrategy

class L2DirectionalClientFilter(ClientFilterStrategy):
    """Filtra clientes com base na distância L2 e na direção do vetor de atualização."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.l2_threshold_global = self.config.get('l2_threshold_global', 1.0)
        self.l2_threshold_peers = self.config.get('l2_threshold_peers', 1.0)
        self.min_rounds = self.config.get('min_rounds', 5)

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> List[Tuple[List[np.ndarray], int]]:
        
        current_round = server_context.get('round', 0)
        previous_global_weights = server_context.get('previous_global_weights')

        if current_round < self.min_rounds:
            self.logger.info(f"Filtro L2: Round {current_round} < {self.min_rounds}, todos os clientes aprovados.")
            return updates

        if previous_global_weights is None:
            self.logger.warning("Filtro L2: Sem pesos globais anteriores, todos os clientes aprovados.")
            return updates

        clean_updates_indices = []
        global_flat = np.concatenate([w.flatten() for w in previous_global_weights])
        all_flat_updates = [
            np.concatenate([layer.flatten() for layer in weights]) 
            for weights, _ in updates
        ]

        for i, (weights, num_examples) in enumerate(updates):
            client_id = client_ids[i]
            current_flat = all_flat_updates[i]

            dot_product = np.dot(current_flat, global_flat)
            if dot_product < 0:
                self.logger.info(f"Cliente {client_id}: REJEITADO (direção invertida, dot={dot_product:.3f})")
                continue

            l2_global = np.linalg.norm(current_flat - global_flat)
            
            peer_updates = all_flat_updates[:i] + all_flat_updates[i+1:]

            if peer_updates:
                peer_mean = np.mean(peer_updates, axis=0)
                l2_peers = np.linalg.norm(current_flat - peer_mean)
            else:
                l2_peers = 0.0

            if l2_global <= self.l2_threshold_global and l2_peers <= self.l2_threshold_peers:
                clean_updates_indices.append(i)
                self.logger.info(f"Cliente {client_id}: ACEITO (l2_global={l2_global:.3f}, l2_peers={l2_peers:.3f})")
            else:
                self.logger.info(f"Cliente {client_id}: REJEITADO (l2_global={l2_global:.3f}, l2_peers={l2_peers:.3f})")

        return [updates[i] for i in clean_updates_indices]


class KrumClientFilter(ClientFilterStrategy):
    """Seleciona um ou 'm' clientes usando o algoritmo Krum/Multi-Krum."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_malicious = self.config.get('num_malicious')
        self.multi_krum = self.config.get('multi_krum', False)
        self.num_to_select = self.config.get('num_to_select')

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> List[Tuple[List[np.ndarray], int]]:
        
        all_weights = [w for w, _ in updates]
        num_models = len(all_weights)
        
        if num_models < 3:
            self.logger.warning(f"Krum: Número insuficiente de modelos ({num_models}), todos aprovados.")
            return updates
        
        num_malicious = self.num_malicious
        if num_malicious is None:
            malicious_percentage = self.config.get('malicious_percentage', 0.2)
            num_malicious = max(1, int(num_models * malicious_percentage))
        
        if num_malicious >= num_models / 2:
            num_malicious = max(1, num_models // 2 - 1)

        flattened_weights = [np.concatenate([w.flatten() for w in weights]) for weights in all_weights]
        distances = np.array([[np.linalg.norm(w1 - w2)**2 for w2 in flattened_weights] for w1 in flattened_weights])
        
        k = num_models - num_malicious - 2
        if k < 1: k = 1
        
        scores = [np.sum(np.sort(distances[i])[1:k+1]) for i in range(num_models)]

        if self.multi_krum:
            m = self.num_to_select or max(1, num_models - num_malicious)
            selected_indices = np.argsort(scores)[:m].tolist()
            self.logger.info(f"Multi-Krum selecionou clientes (índices): {selected_indices}")
        else:
            best_model_idx = int(np.argmin(scores))
            selected_indices = [best_model_idx]
            self.logger.info(f"Krum selecionou cliente (índice): {selected_indices[0]}")

        return [updates[i] for i in selected_indices]


class ClusteringClientFilter(ClientFilterStrategy):
    """Filtra clientes selecionando o maior cluster de modelos."""
    
    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> List[Tuple[List[np.ndarray], int]]:

        all_weights = [w for w, _ in updates]
        num_models = len(all_weights)
        
        if num_models <= 2:
            return updates

        flattened_models = [np.concatenate([w.flatten() for w in model]) for model in all_weights]
        similarity_matrix = np.zeros((num_models, num_models))
        
        for i in range(num_models):
            for j in range(i, num_models):
                if i == j:
                    similarity_matrix[i, j] = 0
                else:
                    norm_i = np.linalg.norm(flattened_models[i])
                    norm_j = np.linalg.norm(flattened_models[j])
                    if norm_i == 0 or norm_j == 0:
                        sim = 1.0 # Distância máxima se um vetor for nulo
                    else:
                        sim = 1.0 - (np.dot(flattened_models[i], flattened_models[j]) / (norm_i * norm_j))
                    similarity_matrix[i, j] = similarity_matrix[j, i] = max(0, min(1, sim))
        
        try:
            condensed_dist = squareform(similarity_matrix)
            Z = linkage(condensed_dist, 'average')
            clusters = fcluster(Z, t=2, criterion='maxclust')
            
            cluster_1_indices = [i for i, c in enumerate(clusters) if c == 1]
            cluster_2_indices = [i for i, c in enumerate(clusters) if c == 2]

            biggest_cluster_indices = cluster_1_indices if len(cluster_1_indices) >= len(cluster_2_indices) else cluster_2_indices
            self.logger.info(f"Clustering selecionou {len(biggest_cluster_indices)} clientes.")
            
            return [updates[i] for i in biggest_cluster_indices]
        except Exception as e:
            self.logger.error(f"Erro no clustering: {e}. Retornando todos os updates.")
            return updates