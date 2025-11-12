import sys
sys.path.append(r'Classification/titanic/model_deep_learning')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
from keras.utils import to_categorical
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"Classification\titanic\model_deep_learning\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def predict(**params):
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])
    # X_val.drop(
    #     columns=config['model_selection']['cols_2_drop'],
    #     inplace=True) 
    # y_val_tf = to_categorical(y_val)      
 
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.h5")
    
    model = keras.models.load_model(model_path)
        
        

    proba = model.predict(X_val)[:, 1]
    preds = np.where(proba > 0.5, 1, 0)
    
    results = pd.DataFrame(y_val).copy()
    results['preds'] = preds
    results['proba'] = proba
    
    acc = {}
    acc['accuracy'] = accuracy_score(y_val, preds)
    acc['f1'] = f1_score(y_val, preds)
    acc['recall'] = recall_score(y_val, preds)
    acc['precision'] = precision_score(y_val, preds)
    acc['roc_auc'] = roc_auc_score(y_val, proba)
    print(acc)
    
    score_path = os.path.join(
        params['reports'],
        "metrics.json")
    
    with open(score_path, 'w') as arquivo:
        json.dump(acc, arquivo)
    
    
    results_path = os.path.join(
        params['predictions'],
        "predictions.parquet")
    
    results.to_parquet(results_path, index=False)
    
if __name__ == "__main__":
        
    
    params = {
        'X_val_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_val']) ,
        'y_val_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),
        'predictions': os.path.join(
            config['init_path'],
            config['output_predict']['path']),
        'model': os.path.join(
            config['init_path'],
            config['model']['path']),
        'model_version': config['model']['model_version']
        }
    print("Begins predict...")
    predict(**params)
    print("data saved...")