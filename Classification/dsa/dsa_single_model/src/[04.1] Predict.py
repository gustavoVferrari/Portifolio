import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/dsa_single_model')
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


yaml_path = r"Classification\dsa_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def predict(**params):
    X_val = pd.read_parquet(params_['X_val_feat_sel'])
    y_val = pd.read_parquet(params_['y_val_feat_sel'])
    X_val.drop(
        columns=config['model_selection']['cols_2_drop'],
        inplace=True) 
    y_val = y_val.astype('int')      
 
    model_path = os.path.join(
        params_['model'],
        f"model_{params_['model_version']}.pkl")
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
        
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    
    results = pd.DataFrame(y_val).copy()
    results['preds'] = preds
    results['proba'] = proba
    
    acc = {}
    acc['accuracy'] = accuracy_score(y_val, preds)
    
    print(acc)
    score_path = os.path.join(
        params_['report'],
        "accuracy.json")
    
    with open(score_path, 'w') as arquivo:
        json.dump(acc, arquivo)
    
    
    results_path = os.path.join(
        params_['predictions'],
        "predictions.parquet")
    
    results.to_parquet(results_path, index=False)
    
if __name__ == "__main__":
        
    
    params_ = {
        'X_val_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val']) ,
        'y_val_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'model': config['model']['path'],
        'report': config['save_reports']['path_reports'],
        'predictions': config['output_predict']['path'],
        'removed_cols': config['save_reports']['path_reports'],
        'model_version': config['model']['model_version']
        }
    print("Begins predict...")
    predict(**params_)
    print("data saved...")