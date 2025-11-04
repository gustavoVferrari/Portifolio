import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_single_model/src')

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import warnings
warnings.filterwarnings('ignore')

# open yaml
yaml_path = r"Regression\house_prices_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    

def feature_forward_selection(**params):
    
    X_train = pd.read_parquet(params['X_train_feat_sel'])
    y_train = pd.read_parquet(params['y_train_feat_sel'])
    
    
    sfs = SFS(
        estimator=LinearRegression(),
        k_features=X_train.shape[1]-1,  
        forward=True,  
        floating=False,  
        scoring='neg_mean_squared_error',
        cv=5, 
        n_jobs=-1
        )    
    
    sfs_pipe = make_pipeline(sfs)
    sfs_fit = sfs_pipe.fit(X_train, y_train)
    
    results = pd.DataFrame.from_dict(sfs_pipe.named_steps['sequentialfeatureselector'].subsets_).T
    
    with open(os.path.join(params['reports'], 'select_forward_results.json'), 'w') as arquivo:
        json.dump(results[['feature_names', 'avg_score']].to_dict(orient='index'), arquivo)
    
    metric_dict = sfs.get_metric_dict()
    num_features = list(metric_dict.keys())
    avg_scores = [metric_dict[k]['avg_score'] for k in num_features]
    std_scores = [metric_dict[k]['std_dev'] for k in num_features]
    cv_scores = [metric_dict[k]['cv_scores'] for k in num_features]  # Scores individuais de cada fold
    
    
    save_path = os.path.join(
        params['save_plot'],
        "select_forward_info.png"
    )
    plt.figure(figsize=(12, 7))
    plt.plot(
        num_features, 
        avg_scores, 
        marker='o', 
        linestyle='-', 
        color='b', 
        linewidth=2,
        markersize=8)

    # Adicionar barras de erro (desvio padrão)
    plt.errorbar(
        num_features, 
        avg_scores, 
        yerr=std_scores, 
        fmt='none', 
        ecolor='red', 
        elinewidth=1, 
        capsize=3, 
        alpha=0.7)

    # Adicionar pontos individuais de cada fold da validação cruzada
    for i, n in enumerate(num_features):
        # Espalhar os pontos horizontalmente para melhor visualização
        x_positions = np.random.normal(n, 0.05, size=len(cv_scores[i]))
        plt.scatter(x_positions, cv_scores[i], color='gray', alpha=0.6, s=30)

    plt.xlabel('qtd of Features', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Select forward selection', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
  
    
if __name__ == "__main__":

    params = {        
        'X_train_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'X_val_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_val_feat_sel':os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'save_plot':config['save_reports']['path_plot'],
        'reports':config['save_reports']['path_reports']
        }
    
    print("Load feature selection:", params)
    feature_forward_selection(**params)
    print("feature selection completed")
    