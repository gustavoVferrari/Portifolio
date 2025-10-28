import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/dsa_single_model')
import json
import numpy as np
import os
import pandas as pd
import yaml

# Carregando o arquivo de configuração
yaml_path = r"Classification\dsa_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_creation(dataset:str, **params):

    df = pd.read_csv(params['raw'])    
     
    dict_cols = dict(columns = list(df.columns))
    
    with open(os.path.join(params['reports'], f'{dataset}_columns.json'), 'w') as arquivo:
        json.dump(dict_cols, arquivo)
    
    print('saving data')    
    df.to_parquet(params['processed'])
    
    
if __name__ == "__main__":
    
    params_train = {
        'raw':os.path.join(
            config['data_gathering']['path'],
            config['data_gathering']['train']),
        'processed':os.path.join(
            config['processed_data']['path'],
            config['processed_data']['train']),
        'reports': config['save_reports']['path_reports']
    }
    params_test = {
        'raw':os.path.join(
            config['data_gathering']['path'],
            config['data_gathering']['test']),
        'processed':os.path.join(
            config['processed_data']['path'],
            config['processed_data']['test']),
        'reports': config['save_reports']['path_reports']
    }
    
    print('Load Feature Creation')
    feature_creation(**params_train, dataset='train')
    feature_creation(**params_test, dataset='test')
    print('Feature Creation completed with sucess')
