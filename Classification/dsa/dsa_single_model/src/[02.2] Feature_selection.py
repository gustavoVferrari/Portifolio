import sys
sys.path.append(r'Classification/dsa/dsa_single_model')

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
yaml_path = r"Classification\dsa\dsa_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_selection_univariate(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])
    
  
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
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
    
    mi = mi.to_dict()        
    report = {
        'numerical_features_2_remove': anova_remove.tolist(),
        'mutual_information':mi
        }
    
    with open(os.path.join(params['reports'],"feature_selection.json"), "w") as f:
        json.dump(report, f)
   
    
if __name__ == "__main__":
    
    params = {        
        'X_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'X_val_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_val_feat_sel':os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'reports': os.path.join(
            config['init_path'],
            config['init_path'],
            config['save_reports']['path_reports']),
        'save_plot': os.path.join(
            config['init_path'],
            config['init_path'],
            config['save_reports']['path_plot'])
        }
    
    print("Load feature selection:", params)
    feature_selection_univariate(**params)
    print("feature selection report completed")
    