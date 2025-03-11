import tensorflow as tf
from typing import Tuple, Any

class ModelFactory:
    """
    Fábrica para criar modelos baseados no tipo especificado.
    """
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """
        Cria e retorna um modelo TensorFlow baseado no tipo especificado.
        
        Args:
            model_name: Nome do modelo a ser criado
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes para classificação
            
        Returns:
            Um modelo TensorFlow compilado
        """
        if model_name == "CNN_SIGN":
            return self._create_cnn_sign(input_shape, num_classes)
        elif model_name == "DNN":
            return self._create_dnn(input_shape, num_classes)
        elif model_name == "VGG":
            return self._create_vgg(input_shape, num_classes)
        elif model_name == "GENERIC_MODEL":
            return self._create_generic_model(input_shape, num_classes)
        else:
            # Modelo padrão se o tipo não for reconhecido
            print(f"Modelo '{model_name}' não reconhecido. Usando modelo genérico.")
            return self._create_generic_model(input_shape, num_classes)
    
    def _create_cnn_sign(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """
        Cria um modelo CNN para o dataset SIGN.
        
        Args:
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes para classificação
            
        Returns:
            Um modelo CNN para o dataset SIGN
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape[1:]))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        # Compilation of the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _create_dnn(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """
        Cria um modelo DNN.
        
        Args:
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes para classificação
            
        Returns:
            Um modelo DNN
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape[1:]),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_vgg(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """
        Cria um modelo baseado em VGG.
        
        Args:
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes para classificação
            
        Returns:
            Um modelo baseado em VGG
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape[1:]),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_generic_model(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """
        Cria um modelo genérico simples.
        
        Args:
            input_shape: Shape dos dados de entrada
            num_classes: Número de classes para classificação
            
        Returns:
            Um modelo genérico simples
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape[1:]),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model