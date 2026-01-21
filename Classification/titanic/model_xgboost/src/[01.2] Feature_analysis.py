import sys
sys.path.append(r"Classification\titanic\model_xgboost")
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
from collections import Counter

# Carregando o arquivo de configuração
yaml_path = r"Classification\titanic\model_xgboost\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_analysis(**params):

    df = pd.read_parquet(params['processed'])       
   
    df.isna().mean().plot.bar(title='missing data')
    path_save = os.path.join(params['save_plot'], 'missing_data.png')
    plt.savefig(path_save, dpi=300, bbox_inches="tight")
    plt.close() 

    categorical_col = list(df.select_dtypes(include=['category','object', 'bool']).columns)    
    numerical_col = list(df.select_dtypes(include=['number']).columns)

    dict_var_type = dict(categorical_var = categorical_col,
                         numerical_var = numerical_col)
    with open(os.path.join(params['reports'], 'cols_type.json'), 'w') as arquivo:
        json.dump(dict_var_type, arquivo)
    
    dict_cardinality = {}
    for col in categorical_col:
        dict_cardinality[col] =  str(df[col].nunique())        
    with open(os.path.join(params['reports'], 'cardinality.json'), 'w') as arquivo:
        json.dump(dict_cardinality, arquivo)
        
    dict_labels_per_cols = {} 
    for col in df.columns:
        dict_labels_per_cols[col] = df[col].nunique()
    with open(os.path.join(params['reports'], 'labels_per_col.json'), 'w') as arquivo:
        json.dump(dict_labels_per_cols, arquivo)      
            
    
if __name__ == "__main__":
    
    params = {      
        'processed':os.path.join(
            config['init_path'],
            config['processed_data']['path'],
            config['processed_data']['train']),
        'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),
        'save_plot': os.path.join(
            config['init_path'],
            config['save_reports']['path_plot']
        )
    }
  
    
    feature_analysis(**params)


