import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic_version_2')

import pandas as pd
import regex
import os
import yaml
import numpy as np


# Carregando o arquivo de configuração
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic_version_2\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_creation(input_data, output_data):

    df = pd.read_csv(input_data)    
  
    print('Criando Features')
    df['Ticket'] = df['Ticket'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['Cabin'] = df['Cabin'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print('Features criadas')

    print('Alterando tipo de dado')
    df = df.astype(
        {
            'Pclass':str, 
            'Age':np.float64, 
            'SibSp':np.float64,
            'Parch':np.float64
            })
    
    print('Salvando dados')    
    df.to_parquet(output_data)
    
    
if __name__ == "__main__":
    input_train=os.path.join(
        config['data_gathering']['path'],
        config['data_gathering']['file_train'])
    
    output_train=os.path.join(
        config['output_data']['path'],
        config['output_data']['file_train'])
    
    input_test=os.path.join(
        config['data_gathering']['path'],
        config['data_gathering']['file_test'])
    
    output_test=os.path.join(
        config['output_data']['path'],
        config['output_data']['file_test'])
    
    print('Load Feature Creation')
    feature_creation(input_train, output_train)
    feature_creation(input_test, output_test)
    print('Feature Creation completed with sucess')
