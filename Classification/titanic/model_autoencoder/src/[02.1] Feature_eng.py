import sys
sys.path.append(r"Classification\titanic\model_autoencoder")
import json
import pandas as pd
import yaml
import pickle
import os
from utils.feat_eng_pipeline import feat_eng_pipeline
from sklearn.model_selection import train_test_split


# Carregando as configurações do arquivo YAML
yaml_path = r"Classification\titanic\model_autoencoder\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def run_feature_eng(**params):
    
    print('Begin Feature Eng')
    df = pd.read_parquet(params['input_data'])
    df.drop(
        columns=params['cols_2_drop'], 
        inplace=True)
    
    print('Split data into train and validation')
    X_train, X_val, y_train, y_val =  train_test_split(
        df.drop(columns=params['target']), 
        df[params['target']],
        test_size=params['val_size'], 
        random_state=params['random_state'])
    
    pipe = feat_eng_pipeline(
        numerical_var=params['num_var'], 
        categorical_var=params['cat_var'])
    
    print('Feature Eng pipe transform')
    pipe.fit(X_train, y_train)
    X_train_trans = pipe.transform(X_train)
    X_val_trans = pipe.transform(X_val)

    print('Save data transform')
    pipe_to_save = os.path.join(
        params['pipe'],
        f'feat_sel_pipe_{params["version"]}.pkl'
        )

    with open(pipe_to_save, 'wb') as arquivo:
        pickle.dump(pipe, arquivo)
    
    X_train_trans.to_parquet(params['output_x_train'])
    X_val_trans.to_parquet(params['output_x_val'])
    
    X_train_trans.columns    
    dict_cols = dict(columns = list(X_train_trans.columns))
    
    with open(os.path.join(params['reports'], 'feat_sel_columns.json'), 'w') as arquivo:
        json.dump(dict_cols, arquivo)
        
    pd.DataFrame(y_train).to_parquet(params['output_y_train'])
    pd.DataFrame(y_val).to_parquet(params['output_y_val'])
    
    print('Feature Eng completed with sucess')
    
    
if __name__ == "__main__":
        
    params = {
        'input_data':os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['input']),                
        'output_x_train' : os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),        
        'output_x_val' : os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),        
        'output_y_train' : os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),        
        'output_y_val' : os.path.join(
             config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),     
        'pipe': os.path.join(
            config['init_path'],
            config['pipe_feat_eng']['path']),   
        'random_state' : config['feat_selection_params']['random_state'],
        'val_size' : config['feat_selection_params']['val_size'],
        'cols_2_drop' : config['feat_selection_params']['cols_2_drop'],
        'num_var' : config['feat_selection_params']['num_var'],
        'cat_var' : config['feat_selection_params']['cat_var'],
        'target' : config['feat_selection_params']['target'],             
        'pipe_version': config['feat_selection_params']['pipe_version'],
        'version' : config['model']['model_version']       
        }
    
    run_feature_eng(**params)