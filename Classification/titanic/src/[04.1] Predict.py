import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic')
import os
import pickle
import pandas as pd
import json
from sklearn.pipeline import make_pipeline
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def predict(**params):
    X_test = pd.read_parquet(params_['X_test_feat_sel'])
    y_test = pd.read_parquet(params_['y_test_feat_sel']) 
    y_test = y_test.astype('int')      
 
    model_path = os.path.join(
        params_['model'],
        f"model.pkl")
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    remove_cols = os.path.join(
        params_['removed_cols'],
        'report.json'
    )
        
    with open(remove_cols, "r") as file:
        cols_2_drop = json.load(file)
        
        
    X_test = X_test.drop(columns=cols_2_drop['removed_variables'])
        
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    results = pd.DataFrame(y_test).copy()
    results['preds'] = preds
    results['proba'] = proba
    
    results_path = os.path.join(
        params_['predictions'],
        "predictions.parquet")
    
    results.to_parquet(results_path, index=False)
    
    return results

if __name__ == "__main__":
        
    X_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_test_file_name'])   
    
    y_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_test_file_name'])
    
    params_ = {
        'X_test_feat_sel': X_test_feat_sel,
        'y_test_feat_sel': y_test_feat_sel,
        'model': config['model']['path'],
        'predictions': config['output_predict']['path'],
        'removed_cols': config['save_reports']['path_reports'],
        }
    
    predict(**params_)