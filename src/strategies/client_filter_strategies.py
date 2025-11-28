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
        self.warmup_rounds = self.config.get('warmup_rounds', 5)
        self.pca_layers = self.config.get('pca_layers', [0, 2, 4]) # Adapte para os índices das camadas Dense
        self.seed = self.config.get('seed', 42)
        self.pca_components = 2
        self.layer_momentum: Dict[int, np.ndarray] = {}

    def filter(
        self, 
        updates: List[Tuple[List[np.ndarray], int]], 
        client_ids: List[int], 
        server_context: Dict[str, Any] = None
    ) -> Tuple[List[Tuple[List[np.ndarray], int]], List[int]]:
        
        current_round = server_context.get('round', 0)
        previous_global_weights = server_context.get('previous_global_weights')

        # --- FASE 1: Warm-up ---
        if current_round < self.warmup_rounds:
            self.logger.info(f"Round {current_round} (Warm-up).")
            return updates, client_ids

        # --- FASE 2: Clustering Dinâmico por Camada ---
        # Tenta encontrar a melhor separação entre honestos e maliciosos
        filtered_updates, filtered_ids = self._dynamic_pca_filter(updates, client_ids, previous_global_weights)

        # Se o filtro PCA não conseguiu separar nada ou sobrou ninguém, retorna vazio
        if not filtered_updates:
            self.logger.warning("Nenhum cliente selecionado no clustering.")
            return [], []

        return filtered_updates, filtered_ids

    def _dynamic_pca_filter(self, updates, client_ids, previous_global_weights):
        if len(updates) < 3:
            self.logger.info("Poucos clientes (<3) para clustering. Mantendo todos.")
            return updates, client_ids

        best_score = -1
        best_layer_idx = -1
        best_labels = None
        best_vectors_matrix = None
        
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
                
                # KMeans (assumindo 2 grupos: Atacantes vs Honestos)
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.seed)
                labels = kmeans.fit_predict(X_pca)
                
                # Silhouette Score (-1 a 1)
                try:
                    score = silhouette_score(X_pca, labels)
                except:
                    score = -1 # Falha se todos forem iguais
                
                if score > best_score:
                    best_score = score
                    best_layer_idx = layer_idx
                    best_labels = labels
                    best_vectors_matrix = X
                
            except Exception as e:
                self.logger.error(f"Erro analisando layer {layer_idx}: {e}")
                continue
        
        # Se a separação for muito ruim (score baixo), fallback para manter todos ou clip
        if best_labels is None or best_score < 0.15:
            self.logger.info(f"Nenhuma separação clara (Score: {best_score:.3f}). Verificando consenso global.")
            
            # Tratamos todos como um único cluster candidato
            all_indices = list(range(len(updates)))
            
            # Usamos a melhor camada encontrada (ou a primeira padrão) para validar
            target_layer = best_layer_idx if best_layer_idx != -1 else self.pca_layers[0]
            
            # Verifica se esse "grande grupo" está alinhado com os pesos globais anteriores
            if self._validate_single_group(updates, all_indices, target_layer, previous_global_weights):
                self.logger.info("Grupo único validado (Alinhado com Pesos Globais Anteriores).")
                return updates, client_ids
            else:
                self.logger.warning("Grupo único REJEITADO (Divergente dos Pesos Globais Anteriores).")
                return [], []
        
        # 2. Separar os índices dos dois clusters
        cluster_0_idxs = [i for i, l in enumerate(best_labels) if l == 0]
        cluster_1_idxs = [i for i, l in enumerate(best_labels) if l == 1]

        # 3. Eleição baseada em Pesos Globais Anteriores
        winning_indices = self._select_winning_cluster(
            best_vectors_matrix, cluster_0_idxs, cluster_1_idxs, best_layer_idx, previous_global_weights
        )

        filtered_updates = [updates[i] for i in winning_indices]
        filtered_ids = [client_ids[i] for i in winning_indices]

        self.logger.info(
            f"Filtro PCA: Layer {best_layer_idx} (Score {best_score:.2f}). "
            f"Cluster Vencedor: {len(filtered_updates)} clientes (de {len(updates)})."
        )
        return filtered_updates, filtered_ids

    def _select_winning_cluster(self, all_vectors_matrix, idxs_0, idxs_1, layer_idx, previous_global_weights):
        """
        Decide qual cluster é o honesto comparando com os pesos globais anteriores.
        """
        global_vec = previous_global_weights[layer_idx].flatten()

        global_round_mean = np.mean(all_vectors_matrix, axis=0)
        global_mean_centered = global_vec - global_round_mean
        
        # Função auxiliar para calcular vetor médio de um grupo de índices
        def get_centered_avg_from_matrix(indices):
            if not indices: return np.zeros_like(global_vec)
            # Seleciona linhas específicas da matriz
            raw_avg = np.mean(all_vectors_matrix[indices], axis=0)
            return raw_avg - global_round_mean

        # Vetores médios dos dois clusters candidatos
        vec_0 = get_centered_avg_from_matrix(idxs_0)
        vec_1 = get_centered_avg_from_matrix(idxs_1)

        # Calcula similaridade de cosseno com o histórico
        # Sim = 1.0 (Alinhado), Sim = -1.0 (Oposto/Ataque)
        sim_0 = 1 - cosine(vec_0, global_mean_centered) if np.any(vec_0) else -1
        sim_1 = 1 - cosine(vec_1, global_mean_centered) if np.any(vec_1) else -1

        self.logger.info(f"Eleição Cluster: C0(n={len(idxs_0)}, sim={sim_0:.3f}) vs C1(n={len(idxs_1)}, sim={sim_1:.3f})")

        # Regra de Decisão:
        # Se ambos positivos, escolhemos o maior
        if sim_0 > 0 and sim_1 > 0:
            self.logger.info("Ambos clusters alinhados. Escolhendo o maior.")
            return idxs_0 if len(idxs_0) >= len(idxs_1) else idxs_1
        
        # Se apenas um for positivo, escolhemos ele
        if sim_0 > 0 and sim_1 < 0:
            return idxs_0
        if sim_1 > 0 and sim_0 < 0:
            return idxs_1

        # Se ambos negativos, rejeitamos
        if sim_0 < -0.1 and sim_1 < -0.1:
            self.logger.warning("Ambos clusters divergem! Rejeitando rodada.")
            return []
    
    def _validate_single_group(self, updates, indices, layer_idx , previous_global_weights):
        """Valida um grupo inteiro contra os pesos globais anteriores (para rounds homogêneos)."""
        global_vec = previous_global_weights[layer_idx].flatten()

        all_vectors_layer = [u[0][layer_idx].flatten() for u in updates]
        global_round_mean = np.mean(all_vectors_layer, axis=0)
        global_mean_centered = global_vec - global_round_mean

        def get_centered_avg_vec(indices):
            if not indices: return np.zeros_like(global_vec)
            raw_vectors = [updates[i][0][layer_idx].flatten() for i in indices]
            raw_avg = np.mean(raw_vectors, axis=0)
            return raw_avg - global_round_mean

        group_vec = get_centered_avg_vec(indices)

        sim = 1 - cosine(group_vec, global_mean_centered)
        self.logger.info(f"Validação Grupo Único: Cosine Sim = {sim:.3f}")
        
        # Se a similaridade for negativa ou muito baixa, é provável que seja um ataque Label Flipping
        # onde todos os clientes estão enviando gradientes invertidos.
        return sim > -0.1
