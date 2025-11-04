import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_single_model')

import pandas as pd
import yaml
import pickle
import os
import json

# Carregando as configurações do arquivo YAML
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Regression/house_prices_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def submission(**params):
     
    print('Data submission')
    df_test = pd.read_parquet(params['test_data'])    
    y_test_id = df_test[['Id']].copy()
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
    
    df_test_transf.columns = df_test_transf.columns.str.replace('num_pipe_1','numerical_pipe')
    df_test_transf.columns = df_test_transf.columns.str.replace('num_pipe_2','numerical_pipe')
    
    # df_test_transf.drop(
    #     columns = params['cols_2_drop_model'],
    #     inplace=True
    # )   
    
    model_path = os.path.join(
        params['model'],
        f"model_{params['model_version']}.pkl")
    # open model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    # report_path = os.path.join(
    #     params['report'],
    #     "report.json")
    # # get cols to predict
    # with open(report_path, "rb") as file:
    #     report = json.load(file)
        
    # predict    
    y_test_id.loc[:,f'{params['target'][0]}'] = model.predict(df_test_transf)
    
    y_test_id.to_csv(
        os.path.join(params['submission'],
                     'submission.csv'),
        index=False)
    
if __name__ == "__main__":
    
      params = {
          'test_data': os.path.join(
              config['processed_data']['path'], 
              config['processed_data']['test']),
          'cols_2_drop_feat_sel' : config['feat_selection_params']['cols_2_drop'],           
          'pipe': config['pipe_feat_eng']['path'],
          'model': config['model']['path'],
          'model_version': config['model']['model_version'],
          'report': config['save_reports']['path_reports'],
          'target':config['feat_selection_params']['target'],
          'cols_2_drop_model':config['model_selection']['cols_2_drop'],
          'target':config['feat_selection_params']['target'],
          'submission':config['submission']['path']
          }
      
      submission(**params)
      
      