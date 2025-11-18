import sys
sys.path.append(r'Classification/dsa/dsa_single_model')
import os
import pickle
import pandas as pd
import json
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    f1_score,     
    roc_auc_score)
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"Classification\dsa\dsa_single_model\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def train_model(**params):
    X_train = pd.read_parquet(params_['X_train_feat_sel'])
    y_train = pd.read_parquet(params_['y_train_feat_sel']) 
    
    X_train.drop(
        columns=config['model_selection']['cols_2_drop'],
        inplace=True)
       
    y_train = y_train.astype('int')
    
    seed_ = params_['random_state']
    
    path = os.path.join(
        params_['model_params'],
        "best_model_params.json")
    
    with open(path, "r", encoding="utf-8") as json_file:
        best_model_params = json.load(json_file)
    
    models = dict(
    rf = [RandomForestClassifier(random_state=seed_)],
    ab = [AdaBoostClassifier(random_state=seed_)],   
    gb = [GradientBoostingClassifier(random_state=seed_)],
    lr = [LogisticRegression(random_state=seed_)],
    ml = [MLPClassifier(random_state=seed_)],
    kn = [KNeighborsClassifier()],
    hg = [HistGradientBoostingClassifier()])
    
    
    seed_ = params_['random_state']
    best_model = list(best_model_params.keys())[0]
    dict_params = list(best_model_params.values())[0]
    clf_model = models[best_model][0]
    clf_model.set_params(**dict_params, random_state=seed_)
    
    pipeline = make_pipeline(
            # PCA(n_components=0.9, svd_solver='full'), 
            clf_model)
    
    print("train model")
    pipeline.fit(X_train, y_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_proba_train = pipeline.predict_proba(X_train)
    
    pd.DataFrame(y_pred_train).to_parquet(
        params_['y_pred_train_path'],
        )    
    pd.DataFrame(y_proba_train).to_parquet(
        params_['y_proba_train_path'],
        )    
    
    print("saving model pkl")
    model_path = os.path.join(
        params_['model'],
        f'model_{params_["model_version"]}.pkl')
        
    with open(model_path, 'wb') as arquivo:
        pickle.dump(pipeline, arquivo)
    print(f"Model saved")
    
    print('Train score:')
    
    dict_score = {}
    dict_score['f1'] = f1_score(y_train, y_pred_train)
    dict_score['accuracy'] = accuracy_score(y_train, y_pred_train)
    dict_score['roc_auc_score'] = roc_auc_score(y_train, y_proba_train[:,1])
    
    dict_score['classificatio_report'] = classification_report(
        y_train,
        y_pred_train,
        output_dict=True)    
    
    print(dict_score)
    
    metrics_path = os.path.join(
        params_['reports'],
        'train_model_metrics.json'
    )
    with open(metrics_path, 'w') as arquivo:
        json.dump(dict_score, arquivo)
    
    
if __name__ == "__main__":
    
    params_ = {        
        'X_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['X_train']),
        'y_train_feat_sel': os.path.join(
            config['init_path'],
            config['feat_selection']['path'],
            config['feat_selection']['y_train']),
        'y_pred_train_path': os.path.join(
            config['init_path'],
            config['train_model']['path'],
            'y_pred_train.parquet'),
        'y_proba_train_path': os.path.join(
            config['init_path'],
            config['train_model']['path'],
            'y_proba_train.parquet'),
        'reports': os.path.join(
            config['init_path'],
            config['save_reports']['path_reports']),
        'model': os.path.join(
            config['init_path'],
            config['model']['path']),        
        'model_params': os.path.join(
            config['init_path'],
            config['train_model']['model_params']),
        'random_state': 23,
        'pipe_version': config['feat_selection_params']['pipe_version'],
        'model_version': config['model']['model_version']
        }
    
    train_model(**params_)