import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import make_pipeline
import yaml

# open yaml
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_selection(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    X_test = pd.read_parquet(params['X_test_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    y_test = pd.read_parquet(params['y_test_feat_sel'])
    
    cols = X_train.filter(like='categorical').columns.tolist()
    
    
    y_train = y_train.astype('int')

    chi_ls = []

    # select only categorical features
    for feature in cols:
        print("Feature:", feature)
        # create contingency table
        arr1 = np.array((y_train.values.flatten()))
        arr2 = np.array(X_train[feature].values.flatten())
        c = pd.crosstab(arr1, arr2)
        
        # chi_test
        p_value = stats.chi2_contingency(c)[1]
        chi_ls.append(p_value)
        
    chi = pd.Series(chi_ls, index=cols)
    
    # Ordenar os valores
    chi_sorted = chi.sort_values(ascending=True)

    # Criar figura
    plt.figure(figsize=(20, 5))

    # Plotar diretamente com matplotlib
    plt.bar(chi_sorted.index, chi_sorted.values, color="skyblue")

    # Ajustar rotação dos rótulos
    plt.xticks(rotation=45)

    # Linha de referência
    plt.axhline(y=0.05, color='r', linestyle='-')

    # Labels e título
    plt.ylabel("p value")
    plt.title("Feature importance based on chi-square test")

    # Caminho de saída
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_chi2.png"
    )

    print("Salvando plot em:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # fecha a figura para evitar sobreposição em loops

    # Remove variables that are NOT significant
    # based on the chi-square test

    remove = chi[chi > 0.05].index

    X_train.drop(remove, axis=1, inplace=True)
    X_test.drop(remove, axis=1, inplace=True)
    
    X_train.to_parquet(params['X_train_feat_sel'])
    X_test.to_parquet(params['X_test_feat_sel'])
    
    print("Variaveis removidas:", remove.tolist())
    print("Novas dimensões do X_train:", X_train.shape)
    print('Variaveis mantidas:', X_train.columns.tolist())
    
    report = {
        'removed_variables': remove.tolist(),
        'new_X_train_shape': X_train.shape,
        'Variaveis mantidas:': X_train.columns.tolist()
        }
    
    with open(os.path.join(params['save_report'],"report.json"), "w") as f:
        json.dump(report, f)
   
    
if __name__ == "__main__":
    X_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train_file_name'])
    
    X_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_test_file_name'])
    
    y_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train_file_name'])
    
    y_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_test_file_name'])
    
    params = {        
        'X_train_feat_sel':X_train_feat_sel,
        'X_test_feat_sel':X_test_feat_sel,
        'y_train_feat_sel':y_train_feat_sel,
        'y_test_feat_sel':y_test_feat_sel,
        'save_plot':config['save_reports']['path_plot'],
        'save_report':config['save_reports']['path_reports']
        }
    
    print("carregando feature selection com os parametros:", params)
    feature_selection(**params)
    print("feature selection finalizado")
    