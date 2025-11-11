import pandas as pd
import yaml
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay


yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
    
    
def model_explicability(**params_):
    
    X_test = pd.read_parquet(params_['X_test_feat_sel'])
    y_test = pd.read_parquet(params_['y_test_feat_sel'])     
    
    
    model_path = os.path.join(
        params_['model'],
        f"model.pkl")    
    with open(model_path, "rb") as file:
            model = pickle.load(file)
    
    s = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
        )
    s = s.sort_values()
    (s
     .plot
     .bar(
        figsize=(20, 8),
        title='Feature Importance'
        )
     )
    
    save_path = os.path.join(
        params_['save_plot'],
        "feature_importance.png"
    )
    print("Salvando em:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
    
    list_feat = s[-3:].index
    
    feature_names_2_explain = []
    feature_names_2_explain_cat = []
    for feat in list_feat:
        if feat.find('numerical') == 0:
            feature_names_2_explain.append(feat)
        elif feat.find('categorical') == 0:
            feature_names_2_explain_cat.append(feat)
            
    cols = list(X_test.columns)
    idx_num = []
    for col in feature_names_2_explain:
        idx_num.append(cols.index(col))
        
    idx_cat = []
    for col in feature_names_2_explain_cat:
        idx_cat.append(cols.index(col))
        
    idx = idx_num + idx_cat
    
    
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title("Parcial dependence plot")

    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_test,
        features = idx,
        categorical_features=idx_cat,
        random_state=23,
        ax=ax
    )
    
    save_path = os.path.join(
        params_['save_plot'],
        "parcial dependence_plot.png"
    )
    print("Salvando em:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title("ICE plot")

    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_test,
        kind='both',
        features = idx_num,
        random_state=23,
        ax=ax
    );
    
    save_path = os.path.join(
        params_['save_plot'],
        "ICE_plot.png"
    )
    print("Salvando em:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
            
    
if __name__ == "__main__":
        
    X_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['X_test_file_name'])   
    
    y_test_feat_sel = os.path.join(
            config['feat_selection']['path'],
            config['feat_selection']['y_test_file_name'])
    
    params_ = {
        'X_test_feat_sel': X_test_feat_sel,
        'y_test_feat_sel': y_test_feat_sel,
        'model': config['model']['path'],
        'save_plot': config['save_reports']['path_plot'],
        'removed_cols': config['save_reports']['path_reports'],
        }
    
    print("Começar processo de predicao...")
    model_explicability(**params_)
    print("Dados previstos com sucesso...")
    
    
    
    
