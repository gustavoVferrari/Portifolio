import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic_voting_model')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
import yaml

# open yaml
yaml_path = r"Classification\titanic_voting_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_selection_univariate(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])
    
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
    chi_sorted = chi.sort_values(ascending=True)
    
    # Plot
    plt.figure(figsize=(20, 5))
    plt.bar(chi_sorted.index, chi_sorted.values, color="skyblue")
    plt.xticks(rotation=90)
    plt.axhline(y=0.05, color='r', linestyle='-')  
    plt.ylabel("p value")
    plt.title("Feature importance based on chi-square test")

    # Caminho de saída
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_chi2.png"
    )

    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # fecha a figura para evitar sobreposição em loops

    # Remove variables that are NOT significant
    # based on the chi-square test

    chi_remove = chi[chi > 0.05].index    
    
    # Anova
    select = X_train.columns.str.contains("numerical")
    cols = X_train.columns
    anova = f_classif(X_train[cols[select]], y_train)
    s = pd.Series(anova[1], index=cols[select])
    s.sort_values(ascending=True).plot.bar(rot=45, figsize=(20, 5))    
    
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_anova.png"
    )
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.ylabel('p value')
    plt.xticks(rotation=90)
    plt.title('Feature importance based on anova')
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
    anova_mask = anova[1] > 0.05
    anova_remove = s[anova_mask].index
    
    # Mutual information    
    mi = mutual_info_classif(X_train, y_train)
    mi = pd.Series(mi)
    mi.index = X_train.columns
    mi.sort_values(ascending=False, inplace=True)
    
    mi.plot.bar(rot=45, figsize=(20, 5))
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_mutual_information.png"
    )
    plt.ylabel('mutual information score')
    plt.title('Feature importance based on mutual information')
    plt.xticks(rotation=90)
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
    
    mi = mi.to_dict()        
    report = {
        'categorical_features_2_remove': chi_remove.tolist(),
        'numerical_features_2_remove': anova_remove.tolist(),
        'mutual_information':mi
        }
    
    with open(os.path.join(params['save_report'],"feature_selection.json"), "w") as f:
        json.dump(report, f)
   
    
if __name__ == "__main__":
    
    params = {        
        'X_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'X_val_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_val_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'save_plot':config['save_reports']['path_plot'],
        'save_report':config['save_reports']['path_reports']
        }
    
    print("Load feature selection:", params)
    feature_selection_univariate(**params)
    print("feature selection report completed")
    