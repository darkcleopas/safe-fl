import os
import numpy as np
import random
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Any, Dict
from PIL import Image
import gc


class DatasetFactory:
    """
    Fábrica para carregar e particionar datasets para Federated Learning.

    Otimizações de performance/memória:
    - Cache global (por processo) dos arrays carregados de disco para evitar
      múltiplas leituras do dataset e duplicação desnecessária de memória.
    - Particionamento por cliente usa slicing sempre que possível (IID) para
      retornar views de NumPy, evitando cópias.
    """

    _CACHE: dict = {}

    def __init__(self):
        pass

    def load_dataset(
        self,
        dataset_name: str,
        client_id: int,
        num_clients: int,
        non_iid: bool = False,
        seed: Optional[int] = None,
        split: str = "train",
        dirichlet_alpha: Optional[float] = None,
        experiment_dir: Optional[str] = None,
        export_distribution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        if dataset_name == "SIGN":
            return self.load_sign(client_id, num_clients, non_iid, split, seed)
        elif dataset_name == "MNIST":
            return self.load_mnist(
                client_id=client_id,
                num_clients=num_clients,
                non_iid=non_iid,
                split=split,
                seed=seed,
                dirichlet_alpha=dirichlet_alpha,
                experiment_dir=experiment_dir,
                export_distribution=export_distribution,
            )
        else:
            print(f"Dataset '{dataset_name}' não reconhecido. Usando MNIST.")
            return self.load_mnist(
                client_id=client_id,
                num_clients=num_clients,
                non_iid=non_iid,
                split=split,
                seed=seed,
                dirichlet_alpha=dirichlet_alpha,
                experiment_dir=experiment_dir,
                export_distribution=export_distribution,
            )

    # ==================== MNIST ====================
    def _ensure_mnist_cached(self, seed: Optional[int]) -> None:
        if 'MNIST' in self._CACHE and 'all' in self._CACHE['MNIST']:
            return

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Normalize and add channel dim
        x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
        x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

        num_classes = 10
        y_train_int = y_train.astype(np.int32)
        y_test_int = y_test.astype(np.int32)
        y_train = tf.keras.utils.to_categorical(y_train_int, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test_int, num_classes)

        self._CACHE.setdefault('MNIST', {})['all'] = {
            'x_train': x_train,
            'y_train': y_train,
            'y_train_int': y_train_int,
            'x_test': x_test,
            'y_test': y_test,
            'y_test_int': y_test_int,
            'num_classes': num_classes,
        }

    def _dirichlet_partition_indices(
        self,
        labels: np.ndarray,
        num_classes: int,
        n_clients: int,
        alpha: float,
    ) -> List[np.ndarray]:
        """Create Dirichlet label-skew partitions returning list of index arrays per client."""
        data_indices = np.arange(len(labels))
        client_indices: List[List[int]] = [[] for _ in range(n_clients)]
        for k in range(num_classes):
            idx_k = data_indices[labels[data_indices] == k]
            np.random.shuffle(idx_k)
            # sample proportions and split
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, split_points)
            for i in range(n_clients):
                client_indices[i].extend(splits[i].tolist())
        # shuffle within client
        for i in range(n_clients):
            np.random.shuffle(client_indices[i])
        return [np.array(ci, dtype=np.int64) for ci in client_indices]

    def _export_distribution_and_plot(
        self,
        experiment_dir: Optional[str],
        dist: Dict[str, List[Dict[int, int]]],
        n_clients: int,
        num_classes: int,
        dataset_name: str,
        alpha: Optional[float],
    ) -> None:
        if experiment_dir is None:
            return
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception:
            # If plotting libs unavailable, skip silently
            return

        out_dir = os.path.join(experiment_dir, 'data_partition')
        os.makedirs(out_dir, exist_ok=True)

        # Build tidy dataframe
        rows = []
        for split in ['train', 'test']:
            for cid in range(n_clients):
                counts = dist[split][cid]
                total = sum(counts.values()) if counts else 0
                for cls in range(num_classes):
                    c = counts.get(cls, 0)
                    frac = (c / total) if total > 0 else 0.0
                    rows.append({
                        'split': split,
                        'client_id': cid + 1,
                        'class': cls,
                        'count': c,
                        'fraction': frac,
                    })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, 'distribution.csv')
        df.to_csv(csv_path, index=False)

        # Single figure with two rows (train/test) stacked bars per client
        fig, axes = plt.subplots(2, 1, figsize=(max(8, n_clients * 0.9), 8), sharex=True)
        fig.suptitle(f"Distribuição por Cliente - {dataset_name} non_iid={alpha is not None} alpha={alpha}")
        cmap = plt.cm.get_cmap('tab10', num_classes)
        for ax_idx, split in enumerate(['train', 'test']):
            ax = axes[ax_idx]
            # build stacked data: matrix [num_classes, n_clients]
            mat = np.zeros((num_classes, n_clients), dtype=float)
            for cid in range(n_clients):
                counts = dist[split][cid]
                total = sum(counts.values()) if counts else 0
                for cls in range(num_classes):
                    mat[cls, cid] = (counts.get(cls, 0) / total) if total > 0 else 0.0
            bottom = np.zeros(n_clients)
            x = np.arange(1, n_clients + 1)
            for cls in range(num_classes):
                ax.bar(x, mat[cls], bottom=bottom, color=cmap(cls), label=f"{cls}" if ax_idx == 0 else None)
                bottom += mat[cls]
            ax.set_ylabel(f"{split.title()} frac")
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis='y', alpha=0.3)
        axes[-1].set_xlabel('Client ID')
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Classe", loc='upper right')
        fig.tight_layout(rect=[0, 0, 0.98, 0.96])
        png_path = os.path.join(out_dir, 'distribution.png')
        plt.savefig(png_path, dpi=200)
        plt.close(fig)

    def load_mnist(
        self,
        client_id: int,
        num_clients: int,
        non_iid: bool = False,
        split: str = "train",
        seed: Optional[int] = None,
        dirichlet_alpha: Optional[float] = None,
        experiment_dir: Optional[str] = None,
        export_distribution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        print(f"Carregando MNIST para cliente {client_id} ({split}) - NON-IID: {non_iid} alpha={dirichlet_alpha}")

        self._ensure_mnist_cached(seed)
        cached = self._CACHE['MNIST']['all']
        x_train_all = cached['x_train']
        y_train_all = cached['y_train']
        y_train_int = cached['y_train_int']
        x_test_all = cached['x_test']
        y_test_all = cached['y_test']
        y_test_int = cached['y_test_int']
        num_classes = cached['num_classes']

        # Build/obtain partition for this configuration
        part_key = ('MNIST', seed, num_clients, bool(non_iid), float(dirichlet_alpha) if dirichlet_alpha is not None else None)
        if 'MNIST' not in self._CACHE:
            self._CACHE['MNIST'] = {}
        if 'partitions' not in self._CACHE['MNIST']:
            self._CACHE['MNIST']['partitions'] = {}

        parts = self._CACHE['MNIST']['partitions'].get(part_key)
        if parts is None:
            # create indices per client
            if non_iid:
                alpha = dirichlet_alpha if dirichlet_alpha is not None else 0.5
                train_idx = self._dirichlet_partition_indices(y_train_int, num_classes, num_clients, alpha)
                test_idx = self._dirichlet_partition_indices(y_test_int, num_classes, num_clients, alpha)
            else:
                # IID: split contiguous blocks
                total_train = len(x_train_all)
                total_test = len(x_test_all)
                base_train = total_train // num_clients
                base_test = total_test // num_clients
                train_idx = []
                test_idx = []
                for i in range(num_clients):
                    ts = i * base_train
                    te = (i + 1) * base_train if i < num_clients - 1 else total_train
                    tr = np.arange(ts, te)
                    tes = i * base_test
                    tee = (i + 1) * base_test if i < num_clients - 1 else total_test
                    tei = np.arange(tes, tee)
                    train_idx.append(tr)
                    test_idx.append(tei)

            # compute distribution stats
            dist: Dict[str, List[Dict[int, int]]] = {'train': [], 'test': []}
            for i in range(num_clients):
                tr_lbls = y_train_int[train_idx[i]]
                te_lbls = y_test_int[test_idx[i]]
                tr_counts = {int(k): int(v) for k, v in zip(*np.unique(tr_lbls, return_counts=True))}
                te_counts = {int(k): int(v) for k, v in zip(*np.unique(te_lbls, return_counts=True))}
                # ensure all classes represented with zero if missing
                tr_counts = {k: tr_counts.get(k, 0) for k in range(num_classes)}
                te_counts = {k: te_counts.get(k, 0) for k in range(num_classes)}
                dist['train'].append(tr_counts)
                dist['test'].append(te_counts)

            parts = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'distribution': dist,
                'exported': False,
            }
            self._CACHE['MNIST']['partitions'][part_key] = parts

        # Export CSV/plot once
        if export_distribution and not parts.get('exported', False):
            try:
                self._export_distribution_and_plot(
                    experiment_dir=experiment_dir,
                    dist=parts['distribution'],
                    n_clients=num_clients,
                    num_classes=num_classes,
                    dataset_name='MNIST',
                    alpha=dirichlet_alpha if non_iid else None,
                )
                parts['exported'] = True
            except Exception as e:
                print(f"[MNIST] Aviso: falha ao exportar distribuição/plot: {e}")

        # Select this client's split
        idx = client_id - 1
        tr_idx = parts['train_idx'][idx]
        te_idx = parts['test_idx'][idx]

        x_train = x_train_all[tr_idx]
        y_train = y_train_all[tr_idx]
        x_test = x_test_all[te_idx]
        y_test = y_test_all[te_idx]

        return x_train, y_train, x_test, y_test, num_classes

    def _ensure_sign_train_cached(self, seed: Optional[int]) -> None:
        if 'SIGN' in self._CACHE and 'train_split' in self._CACHE['SIGN']:
            return

        IMG_HEIGHT = 30
        IMG_WIDTH = 30

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        data_dir = 'data/gtsrb-german-traffic-sign/Train/'
        num_categories = len(os.listdir(data_dir))

        def load_images(directory):
            images = []
            labels = []
            for i in range(num_categories):
                path = os.path.join(directory, str(i))
                if not os.path.exists(path):
                    continue
                for img_file in os.listdir(path):
                    try:
                        img_path = os.path.join(path, img_file)
                        with Image.open(img_path).convert('RGB') as image:
                            resize_image = image.resize((IMG_HEIGHT, IMG_WIDTH))
                            img_array = np.array(resize_image, dtype=np.float16) / 255.0
                            images.append(img_array)
                            labels.append(i)
                    except Exception as e:
                        print(f"Erro ao carregar imagem {img_path}: {e}")
            return np.array(images, dtype=np.float16), np.array(labels)

        x_images, y_labels = load_images(data_dir)

        indices = np.arange(x_images.shape[0])
        np.random.shuffle(indices)
        x_images = x_images[indices]
        y_labels = y_labels[indices]

        x_train, x_test, y_train, y_test = train_test_split(
            x_images, y_labels, test_size=0.3, random_state=42, shuffle=True
        )

        y_train = tf.keras.utils.to_categorical(y_train, num_categories)
        y_test = tf.keras.utils.to_categorical(y_test, num_categories)

        del x_images
        del y_labels
        gc.collect()

        self._CACHE.setdefault('SIGN', {})['train_split'] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'num_classes': num_categories,
        }

    def _ensure_sign_test_cached(self, seed: Optional[int]) -> None:
        if 'SIGN' in self._CACHE and 'test_all' in self._CACHE['SIGN']:
            return

        IMG_HEIGHT = 30
        IMG_WIDTH = 30

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        data_dir = 'data/gtsrb-german-traffic-sign/Test/'
        num_categories = len(os.listdir(data_dir))

        def load_images(directory):
            images = []
            labels = []
            for i in range(num_categories):
                path = os.path.join(directory, str(i))
                if not os.path.exists(path):
                    continue
                for img_file in os.listdir(path):
                    try:
                        img_path = os.path.join(path, img_file)
                        with Image.open(img_path).convert('RGB') as image:
                            resize_image = image.resize((IMG_HEIGHT, IMG_WIDTH))
                            img_array = np.array(resize_image, dtype=np.float16) / 255.0
                            images.append(img_array)
                            labels.append(i)
                    except Exception as e:
                        print(f"Erro ao carregar imagem {img_path}: {e}")
            return np.array(images, dtype=np.float16), np.array(labels)

        x_images, y_labels = load_images(data_dir)
        y_onehot = tf.keras.utils.to_categorical(y_labels, num_categories)

        del y_labels
        gc.collect()

        self._CACHE.setdefault('SIGN', {})['test_all'] = {
            'x_test': x_images,
            'y_test': y_onehot,
            'num_classes': num_categories,
        }

    def load_sign(
        self,
        client_id: int,
        num_clients: int,
        non_iid: bool = False,
        split: str = "train",
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        print(f"Carregando SIGN para cliente {client_id} ({split}) - NON-IID: {non_iid}")

        if split == "train":
            self._ensure_sign_train_cached(seed)
            cached = self._CACHE['SIGN']['train_split']
            x_train_all = cached['x_train']
            y_train_all = cached['y_train']
            x_test_all = cached['x_test']
            y_test_all = cached['y_test']
            num_categories = cached['num_classes']

            if non_iid:
                try:
                    with open(f'data/SIGN/{num_clients}/idx_train_{client_id-1}.pickle', 'rb') as handle:
                        idx_train = pickle.load(handle)
                    with open(f'data/SIGN/{num_clients}/idx_test_{client_id-1}.pickle', 'rb') as handle:
                        idx_test = pickle.load(handle)
                except Exception as e:
                    print(f"Erro ao carregar partição não-IID: {e}")
                    raise e

                x_train = x_train_all[idx_train]
                y_train = y_train_all[idx_train]
                x_test = x_test_all[idx_test]
                y_test = y_test_all[idx_test]
            else:
                x_train, y_train, x_test, y_test = self._partition_data(
                    x_train_all, y_train_all, x_test_all, y_test_all,
                    client_id, num_clients
                )

            return x_train, y_train, x_test, y_test, num_categories

        elif split == "test":
            self._ensure_sign_test_cached(seed)
            cached = self._CACHE['SIGN']['test_all']
            return None, None, cached['x_test'], cached['y_test'], cached['num_classes']

        raise ValueError(f"Split inválido: {split}")

    def _partition_data(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        client_id: int,
        num_clients: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        client_idx = client_id - 1
        train_size = len(x_train)
        test_size = len(x_test)
        train_per_client = train_size // num_clients
        test_per_client = test_size // num_clients

        train_start = client_idx * train_per_client
        train_end = (client_idx + 1) * train_per_client if client_idx < num_clients - 1 else train_size
        test_start = client_idx * test_per_client
        test_end = (client_idx + 1) * test_per_client if client_idx < num_clients - 1 else test_size

        client_x_train = x_train[train_start:train_end]
        client_y_train = y_train[train_start:train_end]
        client_x_test = x_test[test_start:test_end]
        client_y_test = y_test[test_start:test_end]
        return client_x_train, client_y_train, client_x_test, client_y_test

    def _flip_labels(self, labels: np.ndarray) -> np.ndarray:
        flipped_labels = np.copy(labels)
        for i in range(len(labels)):
            if random.random() < 0.5:
                if isinstance(labels[i], np.ndarray):
                    flipped_labels[i][0] = (labels[i][0] + 1) % 10
                else:
                    flipped_labels[i] = (labels[i] + 1) % 10
        return flipped_labels

    def _flip_multiclass_labels(self, one_hot_labels: np.ndarray) -> np.ndarray:
        flipped_labels = np.copy(one_hot_labels)
        num_classes = one_hot_labels.shape[1]
        for i in range(len(one_hot_labels)):
            if random.random() < 0.5:
                current_class = np.argmax(one_hot_labels[i])
                new_class = (current_class + 1) % num_classes
                flipped_labels[i] = np.zeros(num_classes)
                flipped_labels[i][new_class] = 1
        return flipped_labels