import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic')
import os
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,     
    roc_auc_score)
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def train_model(**params):
    X_train = pd.read_parquet(params_['X_train_feat_sel'])
    X_test = pd.read_parquet(params_['X_test_feat_sel'])
    y_train = pd.read_parquet(params_['y_train_feat_sel'])
    y_test = pd.read_parquet(params_['y_test_feat_sel']) 
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    seed_ = params_['random_state']
    
    path = os.path.join(
        params_['model_params'],
        "model_best_params.json")
    
    with open(path, "r", encoding="utf-8") as json_file:
        best_model_params = json.load(json_file)
    
    models = dict(
    rf = [RandomForestClassifier(random_state=seed_)],
    ab = [AdaBoostClassifier(random_state=seed_)],   
    gb = [GradientBoostingClassifier(random_state=seed_)],
    ml = [MLPClassifier(random_state=seed_)],
    hg = [HistGradientBoostingClassifier()])
    
    
    seed_ = params_['random_state']
    best_model = list(best_model_params.keys())[0]
    dict_params = list(best_model_params.values())[0]
    clf_model = models[best_model][0]
    clf_model.set_params(**dict_params, random_state=seed_)
    
    print("Treinando o modelo")
    clf_model.fit(X_train, y_train)
    
    y_pred_train = clf_model.predict(X_train)
    y_proba_train = clf_model.predict_proba(X_train)
    
    pd.DataFrame(y_pred_train).to_parquet(
        params_['y_pred_train_path'],
        )    
    pd.DataFrame(y_proba_train).to_parquet(
        params_['y_proba_train_path'],
        )    
  
    
    print("Salvando pkl do Modelo")
    model_path = os.path.join(
        config['model']['base_path'],
        'model.pkl')
    
    with open(model_path, 'wb') as arquivo:
        pickle.dump(clf_model, arquivo)
    print(f"Modelo salvo com sucesso")
    
    print('score de treino:')
    
    dict_score = {}
    dict_score['f1'] = f1_score(y_train, y_pred_train)
    dict_score['roc_auc_score'] = roc_auc_score(y_train, y_proba_train[:,1])
    
    dict_score['classificatio_report'] = classification_report(
        y_train,
        y_pred_train,
        output_dict=True)
    
    
    metrics_path = os.path.join(
        params_['report'],
        'train_model_metrics.json'
    )
    with open(metrics_path, 'wb') as arquivo:
        json.dump(dict_score, arquivo)
    
    
if __name__ == "__main__":
    X_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train_file_name'])
        
    y_train_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train_file_name'])
    
    y_pred_train_path = os.path.join(
            config['train_model']['path'],
            'y_pred_train.parquet')
    
    y_proba_train_path = os.path.join(
            config['train_model']['path'],
            'y_proba_train.parquet')
    
    params_ = {        
        'X_train_feat_sel': X_train_feat_sel,
        'y_train_feat_sel': y_train_feat_sel,
        'y_pred_train_path': y_pred_train_path,
        'y_proba_train_path': y_proba_train_path,
        'report': config['save_reports']['path_reports'],
        'model': config['model']['path'],
        'random_state': 42,
        }
    
    train_model(**params_)