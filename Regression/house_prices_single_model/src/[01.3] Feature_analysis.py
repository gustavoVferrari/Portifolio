import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_single_model')
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import yaml
from collections import Counter

# Carregando o arquivo de configuração
yaml_path = r"Regression/house_prices_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_analysis(**params):

    df = pd.read_parquet(params['processed'])       
   
    # missing data plot
    df.isna().mean().plot.bar(title='missing data', figsize=(20,10))
    path_save = os.path.join(params['save_plot'], 'missing_data.png')
    plt.savefig(path_save, dpi=300, bbox_inches="tight")
    plt.close() 

    categorical_col = list(df.select_dtypes(include=['category','object', 'bool']).columns)    
    numerical_col = list(df.select_dtypes(include=['number']).columns)

    # saving cols type
    dict_var_type = dict(categorical_var = categorical_col,
                         numerical_var = numerical_col)
    with open(os.path.join(params['reports'], 'cols_type.json'), 'w') as arquivo:
        json.dump(dict_var_type, arquivo)
    
    # cardinality categorical cols
    dict_cardinality = {}
    for col in categorical_col:
        dict_cardinality[col] =  str(df[col].nunique())        
    with open(os.path.join(params['reports'], 'cardinality.json'), 'w') as arquivo:
        json.dump(dict_cardinality, arquivo)
        
    # labels per cols
    dict_labels_per_cols = {} 
    for col in df.columns:
        dict_labels_per_cols[col] = df[col].nunique()
    with open(os.path.join(params['reports'], 'labels_per_col.json'), 'w') as arquivo:
        json.dump(dict_labels_per_cols, arquivo) 
    
    # correlation matrix    
    corr_matrix = df[numerical_col].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))
    mask = mask[1:, :-1]

    plt.figure(figsize=(24,24))
    plt.title("Correlation Matrix")
    sns.heatmap(corr_matrix.iloc[1:,:-1], 
                mask=mask , 
                annot=True, 
                cmap='flare', 
                linewidths=2, 
                square=True);
    path_save = os.path.join(params['save_plot'], 'corr_data.png')
    plt.savefig(path_save, dpi=300, bbox_inches="tight")
    plt.close()      
            
    
if __name__ == "__main__":
    
    params = {      
        'processed':os.path.join(
            config['processed_data']['path'],
            config['processed_data']['train']),
        'reports': config['save_reports']['path_reports'],
        'save_plot': config['save_reports']['path_plot'],
    }
  
    
    feature_analysis(**params)


