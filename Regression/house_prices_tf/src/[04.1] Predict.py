import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_single_model')
import os
import pickle
import pandas as pd
import json
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"Regression/house_prices_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def predict(**params):
    X_val = pd.read_parquet(params_['X_val_feat_sel'])
    y_val = pd.read_parquet(params_['y_val_feat_sel'])
    # X_val.drop(
    #     columns=config['model_selection']['cols_2_drop'],
    #     inplace=True) 
    y_val = y_val.astype('int')      
 
    model_path = os.path.join(
        params_['model'],
        f"model_{params_['model_version']}.pkl")
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
        
    preds = model.predict(X_val)
    
    results = pd.DataFrame(y_val).copy()
    results['preds'] = preds

    
    metric = {}
    metric['rmse'] = root_mean_squared_error(y_val, preds)
    metric['r2'] = r2_score(y_val, preds)
    metric['mape'] = mean_absolute_percentage_error(y_val, preds)
    
    print(metric)
    score_path = os.path.join(
        params_['report'],
        "metric.json")
    
    with open(score_path, 'w') as arquivo:
        json.dump(metric, arquivo)
    
    
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