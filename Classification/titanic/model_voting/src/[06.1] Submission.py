import sys
sys.path.append(r'Classification/titanic/model_voting')

import pandas as pd
import yaml
import pickle
import os
import json

# Carregando as configurações do arquivo YAML
yaml_path = r"Classification\titanic\model_voting\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def submission(**params):
     
    print('Data submission')
    df_test = pd.read_parquet(params['test_data'])    
    y_test_id = df_test[['PassengerId']].copy()
    df_test.drop(
        columns=params['cols_2_drop_feat_sel'], 
        inplace=True)
    # read pipe
    pipe_path = os.path.join(
        params['model'],
        f"feat_sel_pipe_{config['feat_selection_params']['pipe_version']}.pkl")
    
    with open(pipe_path, "rb") as file:
        pipe = pickle.load(file)
    # appply pipe
    df_test_transf = pipe.transform(df_test) 
    
    df_test_transf.drop(
        columns = params['cols_2_drop_model'],
        inplace=True
    )   
    
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.pkl")
    # open model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
        
    # predict    
    y_test_id.loc[:,f'{params['target'][0]}'] = model.predict(df_test_transf)
    
    y_test_id.to_csv(
        os.path.join(params['submission'],
                     'submission.csv'),
        index=False)
    
if __name__ == "__main__":
    
    
    params = {
          'test_data': os.path.join(
              config['init_path'],
              config['processed_data']['path'], 
              config['processed_data']['test']),                 
          'pipe': os.path.join(
            config['init_path'],
            config['pipe_feat_eng']['path']), 
          'model': os.path.join(
              config['init_path'],
              config['model']['path']),
          'model_version': config['model']['model_version'],
          'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),
          'target':config['feat_selection_params']['target'],
          'cols_2_drop_model':config['model_selection']['cols_2_drop'],
          'cols_2_drop_feat_sel' : config['feat_selection_params']['cols_2_drop'],    
          'submission': os.path.join(
              config['init_path'],
              config['submission']['path'])
          }
      
    submission(**params)
      
      