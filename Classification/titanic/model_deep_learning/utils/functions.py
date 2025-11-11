import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def modelo_classificacao_tf(input_shape):
    """
    Cria e retorna um modelo Keras Sequential para regressão.
    
    Args:
        input_shape (int): O número de colunas (features) de entrada.
    """
    # PEP 8 recomenda o uso de parênteses para quebrar linhas longas
    model = Sequential(
        [
            # Primeira camada densa com ativação ReLU (comum)
            Dense(16, activation='relu', input_shape=(input_shape,)),
            # Camada oculta
            Dense(16, activation='relu'),
            # Camada oculta
            Dense(16, activation='relu'),
            # Camada oculta
            Dense(16, activation='relu'),
            # Camada oculta
            Dense(2, activation='sigmoid'),
        ]
    )
    return model