import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic_voting_model')
import json
import numpy as np
import os
import pandas as pd
import yaml

# Carregando o arquivo de configuração
yaml_path = r"Classification\titanic_voting_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_creation(dataset:str, **params):

    df = pd.read_csv(params['raw'])    
  
    print('Run Features')
    df['Ticket'] = df['Ticket'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['Cabin'] = df['Cabin'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['Ticket_1p'] = df['Ticket'].apply(lambda row: row[:1] if pd.notnull(row) else row)
    df['Cabin_1p'] = df['Cabin'].apply(lambda row: row[:1] if pd.notnull(row) else row)
    df['Embarked_mod'] = df['Embarked'].map({'S':'SQ', 'Q':'SQ', 'C':'C'})    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
 
    df = df.astype(
        {
            'Pclass':str, 
            'Age':np.float64, 
            'SibSp':np.float64,
            'Parch':np.float64
            })
    
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
