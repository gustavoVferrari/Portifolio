import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic_version_1')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# open yaml
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic_version_1\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_selection(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    X_val = pd.read_parquet(params['X_val_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    y_val = pd.read_parquet(params['y_val_feat_sel'])
    
    
    sfs = SFS(
        estimator=RandomForestClassifier(random_state=23),
        k_features=X_train.shape[0]-1,  
        forward=True,  
        floating=False,  
        scoring='accuracy',
        cv=3, 
        n_jobs=-1
        )    
    
    sfs_pipe = make_pipeline(sfs)
    sfs_fit = sfs_pipe.fit(X_train, y_train)
    
if __name__ == "__main__":
    X_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train_file'])
    
    X_val_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val_file'])
    
    y_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train_file'])
    
    y_val_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val_file'])
    
    params = {        
        'X_train_feat_sel':X_train_feat_sel,
        'X_val_feat_sel':X_val_feat_sel,
        'y_train_feat_sel':y_train_feat_sel,
        'y_val_feat_sel':y_val_feat_sel,
        'save_plot':config['save_reports']['path_plot'],
        'save_report':config['save_reports']['path_reports']
        }
    
    print("Load feature selection:", params)
    feature_selection(**params)
    print("feature selection completed")
    