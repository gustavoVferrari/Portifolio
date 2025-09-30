import pandas as pd
import regex
import os
import yaml
import numpy as np


yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_creation(path_input, path_output):

    df = pd.read_csv(path_input)    
  
    print('Criando Features')
    df['Ticket'] = df['Ticket'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['Cabin'] = df['Cabin'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    df['Ticket_1p'] = df['Ticket'].apply(lambda row: row[:1] if pd.notnull(row) else row)
    df['Cabin_1p'] = df['Cabin'].apply(lambda row: row[:1] if pd.notnull(row) else row)
    df['Embarked_mod'] = df['Embarked'].map({'S':'SQ', 'Q':'SQ', 'C':'C'})
    df['SibSp_mod'] = df['SibSp'].apply(lambda row: 'yes' if row > 0 else 'no')
    print('Features criadas')

    print('Alterando tipo de dado')
    df = df.astype(
        {
            'Pclass':str, 
            'Survived':'str', 
            'Age':np.float64, 
            'SibSp':np.float64,
            'Parch':np.float64
            }
    )
    print('Salvando dados')
    df.to_parquet(path_output)
    
if __name__ == "__main__":
    path_input=os.path.join(
        config['input']['base_path'],
        config['input']['file_name'])
    
    path_output=os.path.join(
        config['output']['base_path'],
        config['output']['file_name'])
    
    print(path_input)
    feature_creation(path_input, path_output)
    print(path_output)
