import os
import numpy as np
import random
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Any
import cv2
from PIL import Image

class DatasetFactory:
    """
    Fábrica para carregar e particionar datasets para Federated Learning.
    """
    
    def __init__(self):
        """Inicializa a fábrica de datasets."""
        pass
    
    def load_dataset(
        self, 
        dataset_name: str, 
        client_id: int, 
        num_clients: int, 
        non_iid: bool = False,
        seed: Optional[int] = None,
        split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Carrega o dataset especificado e particiona para federated learning.
        
        Args:
            dataset_name: Nome do dataset a ser carregado
            client_id: ID do cliente (0 para o servidor)
            num_clients: Número total de clientes
            non_iid: Se verdadeiro, particiona os dados de forma não-IID
            seed: Seed para reprodutibilidade
            split: "train" ou "test" para carregar partições específicas
            
        Returns:
            Tupla (x_train, y_train, x_test, y_test, num_classes)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        if dataset_name == "SIGN":
            return self.load_sign(client_id, num_clients, non_iid, split)
        else:
            # Dataset padrão se o tipo não for reconhecido
            print(f"Dataset '{dataset_name}' não reconhecido. Usando SIGN.")
            return self.load_sign(client_id, num_clients, non_iid, split)
    
    def load_sign(
        self, 
        client_id: int, 
        num_clients: int, 
        non_iid: bool = False,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Carrega o dataset SIGN (German Traffic Signs) e particiona para federated learning.
        
        Args:
            client_id: ID do cliente (0 para o servidor)
            num_clients: Número total de clientes
            non_iid: Se verdadeiro, particiona os dados de forma não-IID
            split: "train" ou "test" para carregar partições específicas
            is_malicious: Se o cliente é malicioso (para simulação de ataques)
            
        Returns:
            Tupla (x_train, y_train, x_test, y_test, num_classes)
        """
        print(f"Carregando SIGN para cliente {client_id} ({split}) - NON-IID: {non_iid}")
        
        # Dimensões das imagens
        IMG_HEIGHT = 30
        IMG_WIDTH = 30
        
        # Caminho do dataset
        data_dir = 'data/gtsrb-german-traffic-sign' + f'/{split.capitalize()}/'
        
        # Verificar número de classes
        num_categories = len(os.listdir(data_dir))

        # Função para carregar imagens
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
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                            
                        image_fromarray = Image.fromarray(image, 'RGB')
                        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
                        images.append(np.array(resize_image))
                        labels.append(i)
                    except Exception as e:
                        print(f"Erro ao carregar imagem {img_path}: {e}")
            
            return np.array(images), np.array(labels)
        
        # Carregar dados
        x_images, y_labels = load_images(data_dir)

        # Normalizar imagens
        x_images = x_images / 255.0

        if split == "train":
            # Embaralhar
            indices = np.arange(x_images.shape[0])
            np.random.shuffle(indices)
            x_images = x_images[indices]
            y_labels = y_labels[indices]
            
            # Dividir em treino e teste
            x_train, x_test, y_train, y_test = train_test_split(
                x_images, y_labels, test_size=0.3, random_state=42, shuffle=True
            )
            
            # Converter para one-hot encoding
            y_train = tf.keras.utils.to_categorical(y_train, num_categories)
            y_test = tf.keras.utils.to_categorical(y_test, num_categories)
        
            if non_iid:
                try:
                    # Carregar partição não-IID pré-definida
                    pickle_path = f'data/SIGN/{num_clients}/idx_train_{client_id-1}.pickle'
                    with open(pickle_path, 'rb') as handle:
                        idx_train = pickle.load(handle)
                    
                    pickle_path = f'data/SIGN/{num_clients}/idx_test_{client_id-1}.pickle'
                    with open(pickle_path, 'rb') as handle:
                        idx_test = pickle.load(handle)
                except Exception as e:
                    print(f"Erro ao carregar partição não-IID: {e}")
                    raise e
                
                x_train = x_train[idx_train]
                y_train = y_train[idx_train]
                
                x_test = x_test[idx_test]
                y_test = y_test[idx_test]

            else:
                # Particionamento simples (IID)
                x_train, y_train, x_test, y_test = self._partition_data(
                    x_train, y_train, x_test, y_test, client_id, num_clients
                )
        
        elif split == "test":
            x_train, y_train = None, None
            x_test = x_images
            y_test = tf.keras.utils.to_categorical(y_labels, num_categories)

        
        return x_train, y_train, x_test, y_test, num_categories
        
    
    def _partition_data(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        client_id: int, 
        num_clients: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Particiona os dados para um cliente específico.
        
        Args:
            x_train: Dados de treino
            y_train: Rótulos de treino
            x_test: Dados de teste
            y_test: Rótulos de teste
            client_id: ID do cliente
            num_clients: Número total de clientes
            
        Returns:
            Tupla (x_train, y_train, x_test, y_test) para o cliente específico
        """
        # Ajustar client_id para base 0 para particionamento
        client_idx = client_id - 1
        
        # Calcular tamanho das partições
        train_size = len(x_train)
        test_size = len(x_test)
        
        train_per_client = train_size // num_clients
        test_per_client = test_size // num_clients
        
        # Calcular índices iniciais e finais
        train_start = client_idx * train_per_client
        train_end = (client_idx + 1) * train_per_client if client_idx < num_clients - 1 else train_size
        
        test_start = client_idx * test_per_client
        test_end = (client_idx + 1) * test_per_client if client_idx < num_clients - 1 else test_size
        
        # Particionar
        client_x_train = x_train[train_start:train_end]
        client_y_train = y_train[train_start:train_end]
        
        client_x_test = x_test[test_start:test_end]
        client_y_test = y_test[test_start:test_end]
        
        return client_x_train, client_y_train, client_x_test, client_y_test
    
    def _flip_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Inverte rótulos para simular um ataque de label-flipping.
        
        Args:
            labels: Array de rótulos
            
        Returns:
            Array com rótulos invertidos
        """
        flipped_labels = np.copy(labels)
        
        # Flip only a percentage of labels (50%)
        for i in range(len(labels)):
            if random.random() < 0.5:
                if isinstance(labels[i], np.ndarray):
                    # For multi-dimensional labels (e.g., from CIFAR10)
                    flipped_labels[i][0] = (labels[i][0] + 1) % 10
                else:
                    # For single value labels
                    flipped_labels[i] = (labels[i] + 1) % 10
        
        return flipped_labels
    
    def _flip_multiclass_labels(self, one_hot_labels: np.ndarray) -> np.ndarray:
        """
        Inverte rótulos em formato one-hot para simulação de ataques.
        
        Args:
            one_hot_labels: Array de rótulos em formato one-hot
            
        Returns:
            Array com rótulos invertidos
        """
        flipped_labels = np.copy(one_hot_labels)
        num_classes = one_hot_labels.shape[1]
        
        # Flip only a percentage of labels (50%)
        for i in range(len(one_hot_labels)):
            if random.random() < 0.5:
                # Encontrar o índice do valor 1 (classe atual)
                current_class = np.argmax(one_hot_labels[i])
                # Escolher uma classe aleatória diferente
                new_class = (current_class + 1) % num_classes
                
                # Resetar o vetor e definir a nova classe
                flipped_labels[i] = np.zeros(num_classes)
                flipped_labels[i][new_class] = 1
        
        return flipped_labels