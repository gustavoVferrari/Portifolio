from sklearn.pipeline import make_pipeline
from feature_engine.imputation import MeanMedianImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogCpTransformer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.discretisation import DecisionTreeDiscretiser
from utils.functions import MedianByYTransformer
from feature_engine.scaling import MeanNormalizationScaler
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder


def feat_eng_pipeline(
    num_var_1:list,
    num_var_2:list,
    cat_var:list
   ):
    
    # numerical var
    median_var_1 = MeanMedianImputer(
        imputation_method = 'median',
        variables = num_var_1)
    
    
    # numerical var 2
    # log transformation
    log_transf = LogCpTransformer(variables=num_var_2)  
    
    # outlier treatment
    outlier = Winsorizer(variables=num_var_2, capping_method='iqr')
    
    median_var_2 = MeanMedianImputer(
        imputation_method = 'median',
        variables = num_var_2)
    
    num_2_pipe = make_pipeline(
        median_var_2,
        outlier,
        log_transf                     
        )
    
     # nominal var
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = cat_var
        ))

    encoder = make_pipeline(
        OneHotEncoder(
        variables = cat_var, 
        drop_last = True
        ))

    rare_label = make_pipeline(
        RareLabelEncoder(
        variables=cat_var,
        tol=.1,
        n_categories=2
    ))
    
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    
    preprocessor  = ColumnTransformer(
        transformers = [
            ("num_pipe_1", median_var_1, num_var_1),
            ("num_pipe_2", num_2_pipe, num_var_2),
            ("categorical_pipe", categorical_pipe, cat_var)
            ])
    
    pipe = make_pipeline(   
        preprocessor.set_output(transform="pandas")       
        )      
    
    return pipe