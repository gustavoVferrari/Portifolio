
import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,     
    roc_auc_score)
from sklearn.model_selection import GridSearchCV
import yaml

plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

# carregar configurações
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def model_selection(**params): 
    X_train = pd.read_parquet(params_['X_train_feat_sel'])
    X_val = pd.read_parquet(params_['X_val_feat_sel'])
    y_train = pd.read_parquet(params_['y_train_feat_sel'])
    y_val = pd.read_parquet(params_['y_val_feat_sel']) 
    
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    seed_ = params_['random_state']
    
    models = dict(
    rf = [
        RandomForestClassifier(random_state=seed_),
        {'randomforestclassifier__n_estimators':[100, 150, 200, 250],
         'randomforestclassifier__criterion': ['gini', 'entropy'], 
         'randomforestclassifier__max_depth': [None, 5, 10, 20],
         'randomforestclassifier__min_samples_split' :[2,4,6]}
        ]
    ,
    ab = [
        AdaBoostClassifier(random_state=seed_),
        {'adaboostclassifier__n_estimators':[150, 200, 250, 300],
         'adaboostclassifier__learning_rate': [0.01, 0.1, 0.001]}
        ]
    ,
    
    gb = [
        GradientBoostingClassifier(random_state=seed_),
        {'gradientboostingclassifier__n_estimators':[100, 150, 200],
         'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.001]}
        ]
    , 
    
    ml = [
        MLPClassifier(random_state=seed_),
        {'mlpclassifier__hidden_layer_sizes':[10, 20, 30, 50],
         'mlpclassifier__activation': ['relu', 'tanh'],
         'mlpclassifier__learning_rate_init':[0.01, 0.001]
         }
        ],
    hg = [HistGradientBoostingClassifier(),
          {'histgradientboostingclassifier__learning_rate': [0.01, 0.1, 0.001],
           'histgradientboostingclassifier__max_iter': [50, 100, 150]}
    ]
          )
    
    # run moel selection    
    dict_best_model_params = {}
    dict_metrics = {}

    for model in models.keys():
        
        classifier = models[model][0]
        params = models[model][1]
        
        print("testando o modelo:", classifier)
        
        len_ = len([k.split('__')[0] for k in params.keys()][0])
        name_ = [k.split('__')[0] for k in params.keys()][0]

        grid_pipeline = make_pipeline(classifier)
        
        grid = GridSearchCV(
            grid_pipeline,
            scoring=params_['score'],
            param_grid=params, 
            cv=params_['cv'],
            n_jobs=-1)
        
        grid_fit = grid.fit(X_train, y_train)
        
        pred_proba_test = grid.predict_proba(X_val)[:,1]
        pred_test = grid.predict(X_val)
        
        dict_metrics[model] = [
            f1_score(y_val, pred_test), 
            roc_auc_score(y_val, pred_proba_test)
            ] 
        
        # Best params    
        dict_best_params = {}
        for k, v in grid_fit.best_params_.items():
            dict_best_params[k[len_+2:]] = v
            
        dict_best_model_params[model] =  dict_best_params
    
    metrics =  pd.DataFrame(
        dict_metrics, 
        index=['f1', 'roc'])
    
    metrics.to_json(
        os.path.join(params_['reports'], 
                     'model_comparison.json')
        )
    
    best_idx = pd.DataFrame(dict_metrics, index=['f1', 'roc']).loc['roc'].argmax()
    best_idx = pd.DataFrame(dict_metrics, index=['f1', 'roc']).columns[best_idx]
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
    dict_predict_proba={}
    dict_score={}
    
    clf_best_params = models[best_idx][0]
    clf_best_params.set_params(**dict_best_model_params[best_idx])
    print(dict_best_model_params[best_idx])
    (pd.DataFrame
     .from_dict(
         dict_best_model_params[best_idx], 
         orient='index',
         columns=[best_idx])
     .to_json(os.path.join(params_['reports'], 
                           'model_best_params.json')))    
    
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

        dict_train_idx[i] = [train_index]
        dict_val_idx[i] = [val_index]

        model_pipe_stf = make_pipeline(clf_best_params)
        
        dict_fit[i] = model_pipe_stf.fit(
            X_train.iloc[dict_train_idx[i][0]], 
            y_train.iloc[dict_train_idx[i][0]])
        
        
        dict_predict[i] = dict_fit[i].predict(X_train.iloc[dict_val_idx[i][0]])
        
        dict_predict_proba[i] = dict_fit[i].predict_proba(X_train.iloc[dict_val_idx[i][0]])[:,1]
        
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
    print("Salvando em:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close() 
    
    for fold in range(0, params_['cv']):
        tmp = pd.DataFrame(y_train.iloc[dict_val_idx[fold][0]]).assign(escore = dict_predict_proba[fold])
        sns.histplot(data=tmp,x='escore', hue='Survived')
        plt.title(f'Fold : {fold}')
        save_path = os.path.join(
            params_['save_plot'],
            f"separation_plane_{fold}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close() 

    print('Processo concluido com sucesso')
    
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
    
     
    params_ = {        
        'X_train_feat_sel': X_train_feat_sel,
        'X_val_feat_sel': X_val_feat_sel,
        'y_train_feat_sel': y_train_feat_sel,
        'y_val_feat_sel': y_val_feat_sel,
        'reports': config['save_reports']['path_reports'],
        'save_plot': config['save_reports']['path_plot'],
        'score': config['model_selection']['score'],
        'random_state': 42,
        'cv': 5
        }
    
    model_selection(**params_)