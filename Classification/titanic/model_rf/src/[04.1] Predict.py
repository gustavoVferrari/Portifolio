
model_type = 'model_rf'

import sys
sys.path.append(rf"Classification\titanic\{model_type}")
import os
import pickle
import pandas as pd
import json
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

yaml_path = rf"Classification\titanic\{model_type}\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)    
    
def predict(**params):
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])
    X_val.drop(
        columns=config['model_selection']['cols_2_drop'],
        inplace=True) 
    y_val = y_val.astype('int')      
 
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.pkl")
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)        
        
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    
    results = pd.DataFrame(y_val).copy()
    results['preds'] = preds
    results['proba'] = proba
    
    acc = {}
    acc['accuracy'] = accuracy_score(y_val, preds)
    
    
    score_path = os.path.join(
        params['reports'],
        "accuracy.json")
    
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