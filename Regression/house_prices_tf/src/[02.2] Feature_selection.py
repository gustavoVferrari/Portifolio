import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Regression/house_prices_tf')

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml

# open yaml
yaml_path = r"Regression\house_prices_tf\src\config.yaml"
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
    s.sort_values(ascending=True).plot.bar(rot=90, figsize=(20, 5))    
    
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_anova.png")
    
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
    
    mi.plot.bar(rot=90, figsize=(20, 5))
    save_path = os.path.join(
        params['save_plot'],
        "feat_importance_mutual_information.png" )
    
    plt.ylabel('mutual information score')
    plt.title('Feature importance based on mutual information')
    print("Saving plot:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  
    
    
    mi = mi.to_dict()        
    report = {
        # 'categorical_features_2_remove': chi_remove.tolist(),
        'numerical_features_2_remove': anova_remove.tolist(),
        'mutual_information':mi
        }    
    
    select = X_train.columns.str.contains('numerical')
    cols = X_train.columns
    # correlation matrix    
    corr_matrix = X_train[cols[select]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))
    mask = mask[1:, :-1]

    plt.figure(figsize=(24,24))
    plt.title("Correlation Matrix feat selection")
    sns.heatmap(
        corr_matrix.iloc[1:,:-1], 
        mask=mask , 
        annot=True, 
        cmap='flare', 
        linewidths=2, 
        square=True);
    
    path_save = os.path.join(
        params['save_plot'], 
        'corr_data_feat_selection.png'
        )
    plt.savefig(
        path_save, 
        dpi=300, 
        bbox_inches="tight")
    plt.close()
    
    with open(os.path.join(params['save_report'],"feature_selection.json"), "w") as f:
        json.dump(report, f)
        
    # variance_inflation_factor
    X_scaled = pd.DataFrame(
        X_train[cols[select]],
        columns = X_train[cols[select]].columns)  
    
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i)
                       for i in range(X_scaled.shape[1])]
    vif_data.sort_values(by="VIF", ascending=False, inplace=True)
    print(vif_data)
    with open(os.path.join(params['save_report'],"vif.json"), "w") as j:
        json.dump(report, j)
   
    
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
    