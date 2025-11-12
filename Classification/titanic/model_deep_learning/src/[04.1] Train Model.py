import sys
sys.path.append(r'Classification/titanic/model_deep_learning')
import os
import pandas as pd
import json
from utils.functions import modelo_classificacao_tf
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import yaml
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"Classification\titanic\model_deep_learning\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def train_model(**params):
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])      
    
   
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)    
    
    # early_stop = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.001,
    #     patience=12,
    #     verbose=1,
    #     mode='max',
    #     restore_best_weights=True
    #     )
    # Criar o modelo
    input_dim = X_train.shape[1]
    model = modelo_classificacao_tf(input_dim)          
    model.compile(
        optimizer='adam',         
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
        )   
    
    print("train model")     
    # O treinamento deve usar os dados escalonados
    history = model.fit(
        X_train,
        y_train,
        epochs=params['epochs'],  # Número de épocas
        batch_size=32,
        # callbacks=[early_stop],
        validation_split=0.2,  # Usar 10% do treino para validação interna
        verbose=2  # Silencia a saída para o modo de produção
    )
    print("\n### loss metrics ###")
    loss_train, accuracy_train, precision_train, recall_train = model.evaluate(X_train, y_train, verbose=0)
    print(f"loss train: {loss_train:.2f}")
    print(f"acc train: {accuracy_train:.2f}") 
    print(f"prec train: {precision_train:.2f}") 
    print(f"recall train: {recall_train:.2f}")       
    
    print("\n### loss metrics ###")
    loss_val, accuracy_val, precision_val, recall_val = model.evaluate(X_val, y_val, verbose=0)
    print(f"loss train: {loss_val:.2f}")
    print(f"acc train: {accuracy_val:.2f}") 
    print(f"prec train: {precision_val:.2f}") 
    print(f"recall train: {recall_val:.2f}")         
       
    
    
    
    save_path = os.path.join(
        params['save_plot'],
        "train_val_loss.png")
    
    print("Treinamento concluído!")    
    plt.plot(history.history['loss'], label='train Loss ')
    plt.plot(history.history['val_loss'], label='val Loss')
    plt.legend()
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()        
   
    
    print("saving model")
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.h5")    
    model.save(model_path)   
    
if __name__ == "__main__":
    
    params = {        
      'X_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'y_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'X_val_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_val_feat_sel': os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),      
        'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),
        'model': os.path.join(
            config['init_path'],
            config['model']['path']),        
        'model_params': os.path.join(
            config['init_path'],
            config['train_model']['model_params']),
        'save_plot': os.path.join(
            config['init_path'],
            config['save_reports']['path_plot']),
        'model_version': config['model']['model_version'],
        'random_state': 23,
        'epochs': 150,
        'loss': config['model_selection']['loss'],
        'pipe_version': config['feat_selection_params']['pipe_version']        
        }
    
    train_model(**params)