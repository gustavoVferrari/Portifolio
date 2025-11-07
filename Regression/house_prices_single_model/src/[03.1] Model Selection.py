
import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_single_model/src')

import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score    
    )
from sklearn.model_selection import GridSearchCV
import yaml
from sklearn.decomposition import PCA
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

yaml_path = r"Regression\house_prices_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def model_selection(**params): 
    
    X_train = pd.read_parquet(params_['X_train_feat_sel'])
    X_val = pd.read_parquet(params_['X_val_feat_sel'])
    y_train = pd.read_parquet(params_['y_train_feat_sel'])
    y_val = pd.read_parquet(params_['y_val_feat_sel']) 
    
    X_train.drop(
        columns=params_['cols_2_drop'],
        inplace=True)
    X_val.drop(
        columns=params_['cols_2_drop'],
        inplace=True)
    
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    seed_ = params_['random_state']
    
    models = dict(
    rf = [
        RandomForestRegressor(random_state=seed_),
        {'randomforestregressor__n_estimators':[100, 150, 200, 250],
         'randomforestregressor__criterion': ['squared_error'], 
         'randomforestregressor__max_depth': [None, 2, 3, 5, 10],
         'randomforestregressor__min_samples_split' :[None, 2,4,6]
         }
        ]
    ,
    ab = [
        AdaBoostRegressor(random_state=seed_),
        {'adaboostregressor__n_estimators':[150, 200, 250, 300],
         'adaboostregressor__learning_rate': [0.01, 0.1, 0.001]}
        ]
    ,    
    gb = [
        GradientBoostingRegressor(random_state=seed_),
        {'gradientboostingregressor__n_estimators':[100, 150, 200, 250, 300],
         'gradientboostingregressor__learning_rate': [0.01, 0.1, 0.001]}
        ]
    ,
    lr = [
        LogisticRegression(random_state=seed_),
        {'logisticregression__max_iter': [50,100],
         'logisticregression__C': [0.01, 0.1, 1, 10]}
        ],     
    ml = [
        MLPRegressor(random_state=seed_),
        {'mlpregressor__hidden_layer_sizes':[ 50, 70, 100, 120, 150],
         'mlpregressor__activation': ['relu', 'tanh'],
         'mlpregressor__learning_rate_init':[0.1, 0.01, 0.001]
         }
        ],
    hg = [HistGradientBoostingRegressor(),
          {'histgradientboostingregressor__learning_rate': [0.01, 0.1, 0.001],
           'histgradientboostingregressor__max_iter': [50, 100, 150],
           }
    ]
          )
    
    # run moel selection    
    dict_best_model_params = {}
    dict_metrics = {}

    for model in models.keys():
        
        regressor = models[model][0]
        params = models[model][1]
        
        print("Running: ", regressor)
        
        len_ = len([k.split('__')[0] for k in params.keys()][0])
        name_ = [k.split('__')[0] for k in params.keys()][0]

        pca_thres = params_['pca_threshold']
        grid_pipeline = make_pipeline(
            PCA(n_components=pca_thres, svd_solver='full'), 
            regressor)
        
        grid = GridSearchCV(
            grid_pipeline,
            scoring=params_['score'],
            param_grid=params, 
            cv=params_['cv'],
            n_jobs=-1)
        
        grid_fit = grid.fit(X_train, y_train)
        
        pred_test = grid.predict(X_val)
        
        dict_metrics[model] = [
            r2_score(y_val, pred_test),
            root_mean_squared_error(y_val, pred_test),
            mean_absolute_percentage_error(y_val, pred_test)           
            ] 
        
        # Best params    
        dict_best_params = {}
        for k, v in grid_fit.best_params_.items():
            dict_best_params[k[len_+2:]] = v
            
        dict_best_model_params[model] =  dict_best_params
    
    metrics =  pd.DataFrame(
        dict_metrics, 
        index=['r2', 'rmse', 'mape'])
    
    metrics.to_json(
        os.path.join(params_['reports'], 
                     'model_comparison.json')
        )
    
    best_idx = pd.DataFrame(dict_metrics, index=['r2', 'rmse', 'mape' ]).loc['rmse'].argmax()
    best_idx = pd.DataFrame(dict_metrics, index=['r2', 'rmse', 'mape' ]).columns[best_idx]
    print("Best model:", best_idx)
    
    # Cross-Validation with best model
    skf = StratifiedKFold(
        n_splits=params_['cv'], 
        random_state=seed_, 
        shuffle=True)
    
    dict_train_idx={}
    dict_val_idx={}
    dict_fit={}
    dict_predict={}
    dict_score={}
    
    clf_best_params = models[best_idx][0]
    clf_best_params.set_params(**dict_best_model_params[best_idx])
    print(dict_best_model_params[best_idx])
    
    file_dict = os.path.join(
        params_['reports'], 
        'best_model_params.json'
        )
    
    dict_best_params={}
    dict_best_params[best_idx] = dict_best_model_params[best_idx]
    with open(file_dict, 'w', encoding='utf-8') as arquivo:
        json.dump(
            dict_best_params, 
            arquivo,
            ensure_ascii=False, 
            indent=4)          
    
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

        dict_train_idx[i] = [train_index]
        dict_val_idx[i] = [val_index]

        model_pipe_stf = make_pipeline(clf_best_params)
        
        dict_fit[i] = model_pipe_stf.fit(
            X_train.iloc[dict_train_idx[i][0]], 
            y_train.iloc[dict_train_idx[i][0]])
        
        
        dict_predict[i] = dict_fit[i].predict(X_train.iloc[dict_val_idx[i][0]])
        
          
        dict_score[i] = dict_fit[i].score(
            X_train.iloc[dict_val_idx[i][0]], 
            y_train.iloc[dict_val_idx[i][0]])
       
    df_score = pd.DataFrame(
        dict_score.items(), 
        columns=['fold', 'score'])
    
    best_fold = df_score.iloc[df_score.score.argmax()]['fold']
    
    df_score.to_json(
        os.path.join(params_['reports'], 
                     'cv_score.json'))    
    
    save_path = os.path.join(
        params_['save_plot'],
        "cv_score.png"
    )
    
    print(f'score at test dataset {dict_fit[best_fold].score(X_val, y_val)}')
    
    sns.pointplot(
        data=df_score, 
        y='score', 
        x='fold')
    
    plt.title('score per fold')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close() 

    print('Run with success')
    
if __name__ == "__main__":    
     
    params_ = {        
        'X_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'X_val_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_val']),
        'y_train_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_val_feat_sel': os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_val']),
        'cols_2_drop':config['model_selection']['cols_2_drop'],
        'reports': config['save_reports']['path_reports'],
        'save_plot': config['save_reports']['path_plot'],
        'score': config['model_selection']['score'],
        'target':config['feat_selection_params']['target'],
        'random_state': 42,
        'cv': 5,
        'pca_threshold':config['feat_selection_params']['pca_threshold']
        }
    
    model_selection(**params_)