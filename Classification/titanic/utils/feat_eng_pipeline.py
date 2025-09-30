from sklearn.pipeline import make_pipeline, make_union
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler



def feat_eng_pipeline(
    numerical_var:list, 
    categorical_var:list):
    
    
    # numerical var
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_var))
    
    # nominal var
    cat_imputer = make_pipeline(
        CategoricalImputer(
        imputation_method = 'missing',
        fill_value = 'missing', 
        variables = categorical_var
        ))

    encoder = make_pipeline(
        OneHotEncoder(
        variables = categorical_var, 
        drop_last = True
        ))

    rare_label = make_pipeline(
        RareLabelEncoder(
        variables=categorical_var,
        tol=.15,
        n_categories=2
    ))
    
    
    numerical_pipe = make_pipeline(median)
    categorical_pipe = make_pipeline(cat_imputer, rare_label, encoder)
    
    
    preprocessor  = ColumnTransformer(
    transformers = [
    ("numerical_pipe", numerical_pipe, numerical_var), 
    ("categorical_pipe", categorical_pipe, categorical_var),
    ]
    )   
    
    pipe = make_pipeline(
        preprocessor.set_output(transform="pandas"),
        MinMaxScaler().set_output(transform="pandas") 
        )
    
    return pipe