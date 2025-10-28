from sklearn.pipeline import make_pipeline, make_union
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, OrdinalEncoder, WoEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler




def feat_eng_pipeline(
    numerical_var:list
   ):
    
    
    # numerical var
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = numerical_var))
    
      
    
    numerical_pipe = make_pipeline(median)
   
    
    
    preprocessor  = ColumnTransformer(
    transformers = [
    ("numerical_pipe", numerical_pipe, numerical_var)
    ]
    )   
    
    pipe = make_pipeline(
        preprocessor.set_output(transform="pandas"),
        # MinMaxScaler().set_output(transform="pandas"),
        )
    
    return pipe