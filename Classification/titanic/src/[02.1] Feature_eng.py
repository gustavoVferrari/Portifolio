import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic')

import pandas as pd
import yaml
import pickle
import os
from utils.feat_eng_pipeline import feat_eng_pipeline
from sklearn.model_selection import train_test_split


# Carregando as configurações do arquivo YAML
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def run_feature_eng(**params):
    
    print('Iniciando o processo de Feature Eng')
    df = pd.read_parquet(params['input_feat_sel'])
    df.drop(
        columns=params['cols_2_drop'], 
        inplace=True)
    
    print('Dividindo os dados em treino e teste')
    X_train, X_test, y_train, y_test =  train_test_split(
        df.drop(columns=params['target']), 
        df[params['target']],
        test_size=params['test_size'], 
        random_state=params['random_state'])
    
    pipe = feat_eng_pipeline(
        numerical_var=params['num_var'], 
        categorical_var=params['cat_var'])
    
    print('Treinando o pipe de Feature Eng')
    pipe.fit(X_train, y_train)
    X_train_trans = pipe.transform(X_train)
    X_test_trans = pipe.transform(X_test)

    print('Salvando os dados transformados e o pipe')
    pipe_to_save = os.path.join(
        params['pipe'],
        'pipe.pkl'
        )

    with open(pipe_to_save, 'wb') as arquivo:
        pickle.dump(pipe, arquivo)
    
    X_train_trans.to_parquet(params['output_x_train'])
    X_test_trans.to_parquet(params['output_x_test'])
    pd.DataFrame(y_train).to_parquet(params['output_y_train'])
    pd.DataFrame(y_test).to_parquet(params['output_y_test'])
    
    print('Processo de Feature Eng rodado com sucesso')
    
    
if __name__ == "__main__":
    input_feat_sel=os.path.join(
            config['input_feat_selection']['path'],
            config['input_feat_selection']['file_name'])
    
    output_X_train = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_train_file_name'])
    
    output_X_test = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_test_file_name'])
    
    output_y_train = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_train_file_name'])
    
    output_y_test = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_test_file_name'])
    
    params = {
        'input_feat_sel':input_feat_sel,
        'random_state':config['feat_selection_params']['random_state'],
        'test_size':config['feat_selection_params']['teste_size'],
        'cols_2_drop':config['feat_selection_params']['cols_2_drop'],
        'num_var':config['feat_selection_params']['num_var'],
        'cat_var':config['feat_selection_params']['cat_var'],
        'target':config['feat_selection_params']['target'],
        'pipe': config['pipe_feat_eng']['path'],
        'output_x_train':output_X_train,
        'output_x_test':output_X_test,
        'output_y_train':output_y_train,
        'output_y_test':output_y_test,
        }
    
    run_feature_eng(**params)