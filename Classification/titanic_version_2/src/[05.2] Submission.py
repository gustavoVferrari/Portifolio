import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic_version_2')

import pandas as pd
import yaml
import pickle
import os
import json

# Carregando as configurações do arquivo YAML
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic_version_2\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
def submission(**params):
    
    print('Data submission')
    df_test = pd.read_parquet(params['test_data'])    
    y_test_id = df_test[params['passanger_id']].copy()
    df_test.drop(
        columns=params['cols_2_drop'], 
        inplace=True)
    # read pipe
    pipe_path = os.path.join(
        params['model'],
        f"pipe.pkl")
    
    with open(pipe_path, "rb") as file:
        pipe = pickle.load(file)
    # appply pipe
    df_test_transf = pipe.transform(df_test)    
    
    model_path = os.path.join(
        params['model'],
        f"model.pkl")
    # open model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    report_path = os.path.join(
        params['report'],
        "report.json")
    # get cols to predict
    with open(report_path, "rb") as file:
        report = json.load(file)
        
    # predict    
    y_test_id.loc[:,f'{params['target'][0]}'] = model.predict(df_test_transf[report['Variaveis mantidas:']])
    
    y_test_id.to_csv(
        os.path.join(params['submission'],
                     'submission.csv'),
        index=False)
    
if __name__ == "__main__":
    
      params = {
          'test_data': os.path.join(config['output_data']['path'], 
                                    config['output_data']['file_test']),
          'passanger_id' : config['feat_selection_params']['cols_2_drop'],
          'cols_2_drop' : config['feat_selection_params']['cols_2_drop'],           
          'pipe': config['pipe_feat_eng']['path'],
          'model': config['model']['path'],
          'report': config['save_reports']['path_reports'],
          'target':config['feat_selection_params']['target'],
          'submission':config['submission']['path']
          }
      
      submission(**params)
      
      