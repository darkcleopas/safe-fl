import numpy as np
from typing import List, Tuple, Dict, Any, Deque
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, cosine
from sklearn.cluster import KMeans
from collections import deque
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.strategies.abc_strategies import ClientFilterStrategy


class AdaptiveL2ClientFilter(ClientFilterStrategy):
    """
    Filtra clientes com base na distância L2, usando um threshold adaptativo
    calculado a partir da mediana e desvio padrão de uma janela móvel das
    últimas N rodadas. Também verifica a direção do vetor.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Parâmetros para o threshold adaptativo
        self.window_size = self.config.get('window_size', 7)
        self.std_dev_multiplier = self.config.get('std_dev_multiplier', 1.5)
        self.min_rounds_history = self.config.get('min_rounds_history', 5)
        self.history_clipping_quantile = self.config.get('history_clipping_quantile', 0.95)

        # Deques para armazenar o histórico de distâncias (apenas clientes aceitos)
        self.global_distance_history: Deque[float] = deque(maxlen=self.window_size)
        self.peer_distance_history: Deque[float] = deque(maxlen=self.window_size)

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        num_clients = len(updates)
        previous_global_weights = server_context.get('previous_global_weights')

        if num_clients == 0:
            return [], []

        if previous_global_weights is None:
            self.logger.warning("Filtro L2 Adaptativo: Sem pesos globais anteriores, todos os clientes aprovados.")
            return updates, client_ids
        
        # 1. Calcular todas as distâncias primeiro
        global_flat = np.concatenate([w.flatten() for w in previous_global_weights])
        all_flat_updates = [
            np.concatenate([layer.flatten() for layer in weights]) 
            for weights, _ in updates
        ]

        median_of_all_updates = np.median(all_flat_updates, axis=0)

        median_update_vector = median_of_all_updates - global_flat
        norm_median_update = np.linalg.norm(median_update_vector)
        if norm_median_update > 1e-9: # Evitar divisão por zero
            median_update_direction = median_update_vector / norm_median_update
        else:
            median_update_direction = None # Não há direção clara se a mediana não mudou

        # 2. Calcular distâncias e direção para cada cliente
        all_distances = []

        for i, current_flat in enumerate(all_flat_updates):
            client_update_vector = current_flat - global_flat
            
            direction_score = 1.0 # Padrão para aceitar se não for possível calcular
            if median_update_direction is not None:
                direction_score = np.dot(client_update_vector, median_update_direction)

            l2_global = np.linalg.norm(client_update_vector)
            l2_peers = np.linalg.norm(current_flat - median_of_all_updates)
            
            all_distances.append({'l2_global': l2_global, 'l2_peers': l2_peers, 'direction': direction_score})

        # 3. Calcular thresholds adaptativos
        if len(self.global_distance_history) < self.min_rounds_history:
            l2_threshold_global = float('inf')
            l2_threshold_peers = float('inf')
            self.logger.info(f"Filtro L2 Adaptativo: Pouco histórico ({len(self.global_distance_history)} < {self.min_rounds_history}). Usando thresholds permissivos.")
        else:
            median_global = np.median(list(self.global_distance_history))
            std_global = np.std(list(self.global_distance_history))
            l2_threshold_global = median_global + self.std_dev_multiplier * std_global

            median_peers = np.median(list(self.peer_distance_history))
            std_peers = np.std(list(self.peer_distance_history))
            l2_threshold_peers = median_peers + self.std_dev_multiplier * std_peers
            self.logger.info(f"Thresholds adaptativos: l2_global={l2_threshold_global:.3f}, l2_peers={l2_threshold_peers:.3f}")

        # 4. Filtrar clientes
        clean_updates_indices = []
        accepted_global_dists = []
        accepted_peer_dists = []

        for i, dist_info in enumerate(all_distances):
            client_id = client_ids[i]
            l2_g, l2_p, direction = dist_info['l2_global'], dist_info['l2_peers'], dist_info['direction']

            rejection_reason = ""
            if direction < 0:
                rejection_reason = f"direção oposta (score={direction:.3f})"
            elif l2_g > l2_threshold_global:
                rejection_reason = f"distância global muito alta (l2_g={l2_g:.3f} > {l2_threshold_global:.3f})"
            elif l2_p > l2_threshold_peers:
                rejection_reason = f"distância de pares muito alta (l2_p={l2_p:.3f} > {l2_threshold_peers:.3f})"

            if not rejection_reason:
                clean_updates_indices.append(i)
                accepted_global_dists.append(l2_g)
                accepted_peer_dists.append(l2_p)
                self.logger.info(f"Cliente {client_id}: ACEITO (l2_global={l2_g:.3f}, l2_peers={l2_p:.3f}, dir_score={direction:.3f})")
            else:
                self.logger.info(f"Cliente {client_id}: REJEITADO ({rejection_reason})")
        
        # Pega as distâncias de todos os clientes da rodada
        all_round_global_dists = [d['l2_global'] for d in all_distances]
        all_round_peer_dists = [d['l2_peers'] for d in all_distances]
        
        # Limita o impacto de outliers extremos no histórico
        clip_g = np.quantile(all_round_global_dists, self.history_clipping_quantile)
        clip_p = np.quantile(all_round_peer_dists, self.history_clipping_quantile)
        
        self.global_distance_history.extend(np.clip(all_round_global_dists, a_min=0, a_max=clip_g))
        self.peer_distance_history.extend(np.clip(all_round_peer_dists, a_min=0, a_max=clip_p))

        filtered_updates = [updates[i] for i in clean_updates_indices]
        filtered_ids = [client_ids[i] for i in clean_updates_indices]
        return filtered_updates, filtered_ids


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
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        current_round = server_context.get('round', 0)
        previous_global_weights = server_context.get('previous_global_weights')

        if current_round < self.min_rounds:
            self.logger.info(f"Filtro L2: Round {current_round} < {self.min_rounds}, todos os clientes aprovados.")
            return updates, client_ids

        if previous_global_weights is None:
            self.logger.warning("Filtro L2: Sem pesos globais anteriores, todos os clientes aprovados.")
            return updates, client_ids

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

        filtered_updates = [updates[i] for i in clean_updates_indices]
        filtered_ids = [client_ids[i] for i in clean_updates_indices]
        return filtered_updates, filtered_ids


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
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        all_weights = [w for w, _ in updates]
        num_models = len(all_weights)
        
        if num_models < 3:
            self.logger.warning(f"Krum: Número insuficiente de modelos ({num_models}), todos aprovados.")
            return updates, client_ids
        
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

        filtered_updates = [updates[i] for i in selected_indices]
        filtered_ids = [client_ids[i] for i in selected_indices]
        return filtered_updates, filtered_ids


class ClusteringClientFilter(ClientFilterStrategy):
    """Filtra clientes selecionando o maior cluster de modelos."""
    
    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:

        all_weights = [w for w, _ in updates]
        num_models = len(all_weights)
        
        if num_models <= 2:
            return updates, client_ids

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
            
            filtered_updates = [updates[i] for i in biggest_cluster_indices]
            filtered_ids = [client_ids[i] for i in biggest_cluster_indices]
            return filtered_updates, filtered_ids
        except Exception as e:
            self.logger.error(f"Erro no clustering: {e}. Retornando todos os updates.")
            return updates, client_ids


class AdaptiveClusteringClientFilter(ClientFilterStrategy):
    """Defesa híbrida stateless baseada na média de similaridade do round."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.iid_threshold = float(self.config.get('iid_threshold', 0.2))
        self.min_clients = 3
        self.emergency_fraction = 0.25
        self.iid_filter = ClusteringClientFilter()

    def filter(
        self,
        updates: List[Tuple[List[np.ndarray], int]],
        client_ids: List[int],
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        num_clients = len(updates)
        if num_clients < self.min_clients:
            self.logger.info(
                f"Filtro adaptativo: poucos clientes ({num_clients} < {self.min_clients}), mantendo todos."
            )
            return updates, client_ids

        flattened = []
        for weights, _ in updates:
            flattened.append(np.concatenate([layer.flatten() for layer in weights]))
        flattened = np.vstack(flattened)

        norms = np.linalg.norm(flattened, axis=1, keepdims=True)
        normalized = flattened / (norms + 1e-12)

        cosine_sim = np.clip(np.dot(normalized, normalized.T), -1.0, 1.0)
        np.fill_diagonal(cosine_sim, 0.0)

        denominator = num_clients * (num_clients - 1)
        if denominator == 0:
            return updates, client_ids

        avg_similarity = float(np.sum(cosine_sim)) / denominator
        self.logger.info(
            f"Filtro adaptativo: média de similaridade={avg_similarity:.4f}, threshold={self.iid_threshold:.4f}"
        )

        if avg_similarity > self.iid_threshold:
            self.logger.info("Modo IID detectado: delegando ao ClusteringClientFilter original.")
            return self.iid_filter.filter(updates, client_ids, server_context)

        self.logger.info("Modo Non-IID detectado: penalizando grupos muito similares.")
        return self._strategy_similarity_penalty(updates, client_ids, cosine_sim)

    def _strategy_similarity_penalty(
        self,
        updates: List[Tuple[List[np.ndarray], int]],
        client_ids: List[int],
        cosine_sim: np.ndarray
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:

        try:
            max_sims = np.max(cosine_sim, axis=1).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
            labels = kmeans.fit_predict(max_sims)
            centers = kmeans.cluster_centers_.flatten()
            attacker_label = int(np.argmax(centers))
            honest_indices = [i for i, lbl in enumerate(labels) if lbl != attacker_label]
        except Exception as exc:
            self.logger.error(f"Falha na penalização por similaridade ({exc}).")
            honest_indices = []

        if not honest_indices:
            sorted_idx = np.argsort(max_sims.ravel())
            min_to_select = max(1, int(len(updates) * self.emergency_fraction))
            honest_indices = sorted_idx[:min_to_select].tolist()
            self.logger.warning(
                "Filtro adaptativo: fallback selecionando clientes de menor similaridade."
            )

        filtered_updates = [updates[i] for i in honest_indices]
        filtered_ids = [client_ids[i] for i in honest_indices]
        self.logger.info(
            f"Penalização Non-IID aceitou {len(filtered_updates)}/{len(updates)} clientes."
        )
        return filtered_updates, filtered_ids


class DynamicLayerPCAFilter(ClientFilterStrategy):
    """
    Defesa que busca a camada com maior separabilidade (Silhouette Score)
    entre dois clusters e filtra a minoria.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Camadas para inspecionar (focando nos Kernels, pulando bias)
        # Na sua rede: layer_0 (Dense 128), layer_2 (Dense 64), layer_4 (Dense 10)
        self.warmup_rounds = self.config.get('warmup_rounds', 0)
        self.pca_layers = self.config.get('pca_layers', [0, 2, 4]) # Adapte para os índices das camadas Dense
        self.seed = self.config.get('seed', 42)
        self.pca_components = 2
        self.similarity_threshold = 0.3
        self.winners_strategy = "largest_cluster_similarity"

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        self.current_round = server_context.get('round', 0)
        self.previous_global_weights = server_context.get('previous_global_weights')
        if not self.previous_global_weights:
            raise ValueError("DynamicLayerPCAFilter requer 'previous_global_weights' no contexto do servidor.")

        # --- FASE 1: Warm-up ---
        if self.current_round < self.warmup_rounds:
            self.logger.info(f"Round {self.current_round} (Warm-up).")
            return updates, client_ids

        # --- FASE 2: Clustering Dinâmico por Camada ---
        # Tenta encontrar a melhor separação entre honestos e maliciosos
        filtered_updates, filtered_ids = self._dynamic_pca_filter(updates, client_ids)

        # Se o filtro PCA não conseguiu separar nada ou sobrou ninguém, retorna vazio
        if not filtered_updates:
            self.logger.warning("Nenhum cliente selecionado no clustering.")
            return [], []
        
        return filtered_updates, filtered_ids

    def _dynamic_pca_filter(self, updates, client_ids):
        if len(updates) < 3:
            self.logger.info("Poucos clientes (<3) para clustering. Mantendo todos.")
            return updates, client_ids

        best_score = -1
        best_layer_idx = -1
        best_labels = None
        
        # 1. Encontrar a camada que melhor separa os grupos (Melhor Silhouette)
        for layer_idx in self.pca_layers:
            try:
                # Extrai vetor achatado apenas da camada específica
                vectors = [w[layer_idx].flatten() for w, _ in updates if layer_idx < len(w)]
                if not vectors: continue
                                
                X = np.vstack(vectors)

                # Normalização L2
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms == 0] = 1
                X = X / norms

                # PCA
                pca = PCA(n_components=self.pca_components)
                X_pca = pca.fit_transform(X)

                # KMeans
                max_k = min(4, len(updates) - 1)
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.seed)
                    labels = kmeans.fit_predict(X_pca)
                    
                    # Silhouette Score (-1 a 1)
                    score = silhouette_score(X_pca, labels)

                    # Bonificação para K=2 (navalha de occam), mas permite K>2 se a separação for muito melhor
                    if score > best_score:
                        best_score = score
                        best_layer_idx = layer_idx
                        best_labels = labels
                        best_k = k
                
            except Exception as e:
                self.logger.error(f"Erro analisando layer {layer_idx}: {e}")
                continue
        
        # Se não houver separação clara (score muito baixo), assume que todos são honestos ou ataque falhou
        if best_score < 0.1: 
            self.logger.info(f"Baixa separação (Score {best_score:.2f}). Aceitando todos.")
            return updates, client_ids
        
        self.logger.info(f"Melhor separação: Layer {best_layer_idx}, K={best_k}, Score={best_score:.2f}")
        
        # 2. Agrupa índices por cluster
        clusters_idxs = {}
        for k in range(best_k):
            clusters_idxs[k] = [i for i, l in enumerate(best_labels) if l == k]

        # 3. Eleição baseada em Pesos Globais Anteriores
        winning_indices = self._select_winning_indices(
            updates, clusters_idxs, best_layer_idx
        )

        filtered_updates = [updates[i] for i in winning_indices]
        filtered_ids = [client_ids[i] for i in winning_indices]

        self.logger.info(
            f"Filtro PCA: Layer {best_layer_idx}, K={best_k}, Score {best_score:.2f}. "
            f"Atualizações Vencedoras: {len(filtered_updates)} clientes (de {len(updates)})."
        )
        return filtered_updates, filtered_ids

    def _select_winning_indices(self, updates, clusters_idxs, layer_idx):
        """
        Decide quais clusters são os honestos de acordo com a estratégia de seleção.
        """
        self.logger.info(f"Estratégia de Seleção: {self.winners_strategy}")
        global_w_prev = self.previous_global_weights[layer_idx].flatten()

        # Se não temos histórico de momentum, não podemos julgar a direção.
        # Fallback: Aceitar o maior cluster (suposição de maioria honesta)
        # if self.last_global_momentum is None or layer_idx not in self.last_global_momentum:
        #     self.logger.warning("Sem momentum histórico para esta layer. Escolhendo o maior cluster.")
        #     largest_cluster_k = max(clusters_idxs, key=lambda k: len(clusters_idxs[k]))
        #     return clusters_idxs[largest_cluster_k]
        
        # Seleciona o maior cluster desde que ele não perca a direção histórica
        if self.winners_strategy == "largest_cluster_similarity":
            largest_cluster_k = max(clusters_idxs, key=lambda k: len(clusters_idxs[k]))
            self.logger.info(f"Maior cluster: {largest_cluster_k} (n={len(clusters_idxs[largest_cluster_k])})")

            # 1. Calcular o peso médio deste cluster
            cluster_weights_list = [updates[i][0][layer_idx].flatten() for i in clusters_idxs[largest_cluster_k]]
            cluster_mean_w = np.mean(cluster_weights_list, axis=0)

            # 2. Calcular o vetor de atualização deste cluster (Delta)
            # Delta = W_cluster_t - W_global_{t-1}
            cluster_update_vec = cluster_mean_w - global_w_prev

            # 3. Cosseno Similaridade com o Histórico
            dist = cosine(global_w_prev, cluster_update_vec) if np.any(cluster_update_vec) else 1.0
            similarity = 1.0 - dist

            self.logger.info(f"Cluster {largest_cluster_k}: Similaridade Cosine = {similarity:.3f} | Threshold = {self.similarity_threshold:.3f}")
            return clusters_idxs[largest_cluster_k]
            # if similarity >= self.similarity_threshold:
            #     self.logger.info(f"Cluster {largest_cluster_k} aceito pela similaridade.")
            #     return clusters_idxs[largest_cluster_k]
            # else:
            #     self.logger.info(f"Cluster {largest_cluster_k} rejeitado pela similaridade. Rejeitando todos.")
            #     return []

        elif self.winners_strategy == "clusters_similarity":
            # Vetor de referência: Direção da atualização global passada
            accepted_indices = []

            # Iteramos sobre TODOS os clusters encontrados (k=2, 3 ou 4)
            for k, idxs in clusters_idxs.items():
                if not idxs: continue
                
                # 1. Calcular o peso médio deste cluster
                cluster_weights_list = [updates[i][0][layer_idx].flatten() for i in idxs]
                cluster_mean_w = np.mean(cluster_weights_list, axis=0)

                # 2. Calcular o vetor de atualização deste cluster (Delta)
                # Delta = W_cluster_t - W_global_{t-1}
                cluster_update_vec = cluster_mean_w - global_w_prev

                # 3. Cosseno Similaridade com o Histórico
                # 1.0 = Mesmo sentido, 0.0 = Ortogonal, -1.0 = Oposto
                # Nota: 1 - cosine dá a distância. Queremos a similaridade (1 - dist) ou produto escalar.
                # A função scipy cosine retorna a DISTÂNCIA (0 a 2).
                dist = cosine(cluster_update_vec, history_vec) if np.any(cluster_update_vec) else 1.0
                similarity = 1.0 - dist
                
                self.logger.info(f"Cluster {k} (n={len(idxs)}): Similaridade Cosine = {similarity:.3f} | Threshold = {self.similarity_threshold:.3f}")

                # Regra de Decisão:
                # > 0: A atualização vai na "mesma direção" geral que o treino vinha seguindo.
                # < 0: A atualização quer reverter o aprendizado anterior (típico de ataques ou dados muito ruidosos).
                if similarity >= self.similarity_threshold:
                    accepted_indices.extend(idxs)
                else:
                    self.logger.info(f"Cluster {k} rejeitado (direção oposta).")

            # Fallback de segurança: Se todos forem rejeitados (ex: mudança drástica de conceito),
            # talvez devamos aceitar o que tem maior similaridade (menos negativo) ou rejeitar tudo.
            # Aqui, vamos garantir que a lista seja ordenada e única
            return sorted(list(set(accepted_indices)))
        else:
            self.logger.warning(f"Estratégia desconhecida: {self.winners_strategy}. Aceitando todos.")
            return [i for i in range(len(updates))]


class PCAGeometricMedianDistanceFilter(ClientFilterStrategy):
    """
    Abordagem: Em vez de clusterizar (que falha em Non-IID pois honestos não clusterizam),
    usamos PCA para projetar em baixa dimensão e calculamos a Mediana Geométrica (ou Component-wise Median).
    Eliminamos os pontos mais distantes desse centro robusto
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pca_layers = self.config.get('pca_layers', [0, 2, 4]) 
        self.n_components = self.config.get('n_components', 5)
        self.warmup_rounds = self.config.get('warmup_rounds', 0)
        self.discard_fraction = self.config.get('discard_fraction', 0.4)

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        self.seed = server_context.get('seed', 42)
        
        current_round = server_context.get('round', 0)
        if current_round < self.warmup_rounds:
            return updates, client_ids

        # 1. Extração (Mesma lógica sua)
        client_vectors = []
        valid_indices = []
        for i, (weights, _) in enumerate(updates):
            layers = [weights[idx].flatten() for idx in self.pca_layers if idx < len(weights)]
            if layers:
                client_vectors.append(np.concatenate(layers))
                valid_indices.append(i)

        if len(client_vectors) < 3:
            return updates, client_ids

        X = np.vstack(client_vectors)
        
        # 2. PCA
        n_comps = min(self.n_components, len(client_vectors) - 1)
        pca = PCA(n_components=n_comps, random_state=self.seed)
        X_reduced = pca.fit_transform(X) # PCA centraliza internamente pela média (ainda é um risco, mas menor na dimensão reduzida)

        # 3. A Mágica: Coordinate-wise Median no Espaço Latente
        # Em vez de média, pegamos a mediana de cada componente principal.
        # Label Flipping costuma jogar os valores para extremos. A mediana ignora extremos.
        robust_centroid = np.median(X_reduced, axis=0)

        # 4. Calcular Distância de cada cliente até o Centro Robusto
        distances = np.linalg.norm(X_reduced - robust_centroid, axis=1)

        # 5. Filtragem baseada em Distância
        # Mantemos os (1 - discard_fraction) clientes mais próximos do centro
        num_keep = int(len(client_vectors) * (1 - self.discard_fraction))
        num_keep = max(num_keep, 2) # Segurança mínima

        # Pega os índices dos 'num_keep' menores valores de distância
        sorted_indices = np.argsort(distances)
        keep_local_indices = sorted_indices[:num_keep]
        
        # Log para debug visual
        self.logger.info(f"R{current_round}: Mantendo {num_keep}/{len(client_vectors)}. Distâncias Max Aceita: {distances[keep_local_indices[-1]]:.2f}, Min Rejeitada: {distances[sorted_indices[num_keep]]:.2f}")

        final_indices = [valid_indices[i] for i in keep_local_indices]
        
        return [updates[i] for i in final_indices], [client_ids[i] for i in final_indices]


class PCAGeometricMedianDirectionFilter(ClientFilterStrategy):
    """
    Abordagem: Em vez de clusterizar (que falha em Non-IID pois honestos não clusterizam),
    usamos PCA para projetar em baixa dimensão e calculamos a Mediana Geométrica (ou Component-wise Median).
    Eliminamos os pontos mais distantes desse centro robusto
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pca_layers = self.config.get('pca_layers', [0, 2, 4]) 
        self.n_components = self.config.get('n_components', 5)
        self.warmup_rounds = self.config.get('warmup_rounds', 0)
        self.selection_strategy = self.config.get('selection_strategy', 'min_cosine') # 'min_cosine' ou 'discard_fraction'
        self.min_cosine = self.config.get('min_cosine', 0.4)
        self.discard_fraction = self.config.get('discard_fraction', 0.4)

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        self.seed = server_context.get('seed', 42)
        
        current_round = server_context.get('round', 0)
        if current_round < self.warmup_rounds:
            return updates, client_ids

        # 1. Extração (Mesma lógica sua)
        client_vectors = []
        valid_indices = []
        for i, (weights, _) in enumerate(updates):
            layers = [weights[idx].flatten() for idx in self.pca_layers if idx < len(weights)]
            if layers:
                client_vectors.append(np.concatenate(layers))
                valid_indices.append(i)

        if len(client_vectors) < 3:
            return updates, client_ids

        X = np.vstack(client_vectors)
        
        # 2. PCA
        n_comps = min(self.n_components, len(client_vectors) - 1)
        pca = PCA(n_components=n_comps, random_state=self.seed)
        X_reduced = pca.fit_transform(X) # PCA centraliza internamente pela média (ainda é um risco, mas menor na dimensão reduzida)

        # 3. A Mágica: Coordinate-wise Median no Espaço Latente
        # Em vez de média, pegamos a mediana de cada componente principal.
        # Label Flipping costuma jogar os valores para extremos. A mediana ignora extremos.
        robust_vector = np.median(X_reduced, axis=0)

        # 4. Filtragem Baseada em Cosseno (Direção)
        # cosseno = 1 - distância_cosseno (scipy retorna distância 0..2)
        # Queremos similaridade: 1.0 (igual), 0.0 (ortogonal), -1.0 (oposto)
        
        kept_indices_local = []
        scores = []

        for i, vec in enumerate(X_reduced):
            # cosine() do scipy calcula a DISTÂNCIA (Dissimilaridade). 
            # Sim = 1 - dist.
            if np.linalg.norm(vec) < 1e-9:
                sim = 0 # Vetor nulo não contribui
            else:
                sim = 1.0 - cosine(vec, robust_vector)
            
            scores.append(sim)

            # Aceitamos se apontar para o mesmo hemisfério que a mediana
            if sim > self.min_cosine:
                kept_indices_local.append(i)

        if self.selection_strategy == 'discard_fraction':
            self.logger.info(f"Seleção por discard_fraction={self.discard_fraction}.")
            # Alternativa: Seleciona os (1 - discard_fraction) mais alinhados
            num_keep = int(len(client_vectors) * (1 - self.discard_fraction))
            num_keep = max(num_keep, 2) # Segurança mínima

            sorted_indices = np.argsort(scores)[::-1] # Ordem decrescente
            kept_indices_local = sorted_indices[:num_keep]
        else:
            self.logger.info(f"Seleção por min_cosine={self.min_cosine}.")
        
        final_indices = [valid_indices[i] for i in kept_indices_local]
        
        # Logs para diagnóstico
        avg_score = np.mean(scores)
        
        # Log para debug visual
        self.logger.info(f"R{current_round}: Cosseno Médio {avg_score:.2f}. Mantendo {len(final_indices)}/{len(client_vectors)} clientes alinhados.")

        return [updates[i] for i in final_indices], [client_ids[i] for i in final_indices]