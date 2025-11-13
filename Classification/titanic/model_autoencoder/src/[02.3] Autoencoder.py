import sys
sys.path.append(r'Classification\titanic\model_autoencoder')

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model

# open yaml
yaml_path = r"Classification\titanic\model_autoencoder\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

import warnings
warnings.filterwarnings('ignore')

  
def feature_autoencoder(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    
    # Construir um Autoencoder
    def build_autoencoder(input_dim, encoding_dim=8):
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder
    
    
    input_dim = X_train.shape[1]  
    autoencoder, encoder = build_autoencoder(input_dim)
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=150,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val))
    
    save_path = os.path.join(
        params['save_plot'],
        "autoencoder_train_loss.png")
    
    print("Treinamento concluído!")    
    plt.plot(history.history['loss'], label='train Loss ')
    plt.plot(history.history['val_loss'], label='val Loss')
    plt.legend()
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    encoded_features_train = encoder.predict(X_train)
    encoded_features_val = encoder.predict(X_val)
    
    pd.DataFrame(encoded_features_train).to_parquet(params['X_train_autoencoder'])
    pd.DataFrame(encoded_features_val).to_parquet(params['X_val_autoencoder']) 
    
    
    print("saving model")
    model_path = os.path.join(
        params['model'],
        f"encoder.h5")    
   
    encoder.save(model_path)           
  
    
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
        'X_train_autoencoder' : os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_train_encoder']),
        'X_val_autoencoder' : os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_val_encoder']),
        'save_plot': os.path.join(
            config['init_path'],
            config['save_reports']['path_plot']), 
        'model': os.path.join(
            config['init_path'],
            config['model']['path'])
        }
    
    print("Load feature selection:", params)
    feature_autoencoder(**params)
    print("feature selection completed")
