import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_tf')

import os
import pandas as pd
import yaml
import numpy as np  # Convenção para NumPy (PEP 8)
from utils.functions import modelo_regressao_tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# open yaml
yaml_path = r"Regression\house_prices_tf\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
def Model_train(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])    
    
    
    # Criar o modelo
    input_dim = X_train.shape[1]
    model = modelo_regressao_tf(input_dim)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mse', 'mae'])
    
    # Visualizar a arquitetura do modelo
    print("### Arquitetura do Modelo ###")
    model.summary()    
    print("Iniciando o treinamento...")

    # O treinamento deve usar os dados escalonados
    history = model.fit(
        X_train,
        y_train,
        epochs=params['epochs'],  # Número de épocas
        batch_size=32,
        validation_split=0.15,  # Usar 10% do treino para validação interna
        verbose=2  # Silencia a saída para o modo de produção
    )
    save_path = os.path.join(
        params['save_plot'],
        "train_val_loss.png"
    )
    print("Treinamento concluído!")    
    plt.plot(history.history['loss'], label='train Loss ')
    plt.plot(history.history['val_loss'], label='val Loss')
    plt.legend()
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
    # --- 5. Avaliação do Modelo ---

    print("\n### train metrics ###")
    # avalia_resultado = [loss, metric1, metric2, ...]
    loss_t, mae_t, mse_t = model.evaluate(X_train, y_train, verbose=0)

    print(f"Loss (MSE) train: {loss_t:.4f}")
    print(f"MAE train: {mae_t:.4f}")
    print(f"RMSE train: {np.sqrt(mse_t):.4f}")
    
    print("\n### loss metrics ###")
    # avalia_resultado = [loss, metric1, metric2, ...]
    loss_v, mae_v, mse_v = model.evaluate(X_val, y_val, verbose=0)

    print(f"Loss (MSE) train: {loss_v:.4f}")
    print(f"MAE train: {mae_v:.4f}")
    print(f"RMSE train: {np.sqrt(mse_v):.4f}")
    
    print("saving model")
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.h5")
    
    model.save(model_path)


if __name__ == "__main__":
        
    
    params = {
        'X_train_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'X_val_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_val_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'model': config['model']['path'],
        'report': config['save_reports']['path_reports'],
        'predictions': config['output_predict']['path'],
        'removed_cols': config['save_reports']['path_reports'],
        'model_version': config['model']['model_version'],
        'save_plot':config['save_reports']['path_plot'],
        'epochs': 30
        }
    print("Begins predict...")
    Model_train(**params)
    print("data saved...")
