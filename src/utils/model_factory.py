import tensorflow as tf
from typing import Tuple, Any

class ModelFactory:
    """
    Fábrica para criar modelos baseados no tipo especificado.
    """
    
    def create_model(self, model_name: str, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
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
            return self._create_cnn_sign(input_shape, num_classes, learning_rate)
        elif model_name == "DNN":
            return self._create_dnn(input_shape, num_classes, learning_rate)
        elif model_name == "VGG":
            return self._create_vgg(input_shape, num_classes, learning_rate)
        elif model_name in ("RESNET18", "RESNET18_MNIST"):
            return self._create_resnet18(input_shape, num_classes, learning_rate)
        elif model_name == "GENERIC_MODEL":
            return self._create_generic_model(input_shape, num_classes, learning_rate)
        else:
            # Modelo padrão se o tipo não for reconhecido
            print(f"Modelo '{model_name}' não reconhecido. Usando modelo genérico.")
            return self._create_generic_model(input_shape, num_classes, learning_rate)
    
    def _create_cnn_sign(self, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
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
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        # Compilation of the model
        lr = learning_rate if learning_rate is not None else 1e-4
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _create_dnn(self, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
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
            # tf.keras.layers.Dense(512, activation='relu'),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        # Compile to ensure training works regardless of caller
        lr = learning_rate if learning_rate is not None else 1e-4
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    def _create_vgg(self, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
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
        # Compile
        lr = learning_rate if learning_rate is not None else 1e-4
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    def _create_generic_model(self, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
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
        lr = learning_rate if learning_rate is not None else 1e-3
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    # ==================== ResNet-18 ====================
    def _create_resnet18(self, input_shape: Tuple, num_classes: int, learning_rate: float | None = None) -> tf.keras.Model:
        """Build a lightweight ResNet-18 compatible with grayscale MNIST.

        Input shape provided is the data batch shape; we strip batch dim.
        """
        xshape = input_shape[1:]
        inputs = tf.keras.Input(shape=xshape)

        def conv_bn_relu(x, filters, kernel_size, strides=1, padding='same'):
            x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                                       use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return tf.keras.layers.ReLU()(x)

        def basic_block(x, filters, stride=1):
            shortcut = x
            x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                                       kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False,
                                       kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Projection if shape changes
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False,
                                                 kernel_initializer='he_normal')(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.ReLU()(x)
            return x

        # Stem
        x = conv_bn_relu(inputs, 64, 3, strides=1)
        # Stages (2,2,2,2) blocks
        x = basic_block(x, 64, stride=1)
        x = basic_block(x, 64, stride=1)

        x = basic_block(x, 128, stride=2)
        x = basic_block(x, 128, stride=1)

        x = basic_block(x, 256, stride=2)
        x = basic_block(x, 256, stride=1)

        x = basic_block(x, 512, stride=2)
        x = basic_block(x, 512, stride=1)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name='ResNet18')
        lr = learning_rate if learning_rate is not None else 1e-3
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model