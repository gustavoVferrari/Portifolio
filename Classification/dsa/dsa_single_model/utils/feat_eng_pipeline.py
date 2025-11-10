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
from sklearn.impute import KNNImputer


def feat_eng_pipeline(
    num_var_1:list,
    num_var_2:list
   ):
    
    
    # numerical var
    median = make_pipeline(
        MeanMedianImputer(
        imputation_method = 'median',
        variables = num_var_2))
    
    
    
    inputer = KNNImputer()    
    
    inputer_pipe = make_pipeline(inputer)  

    preprocessor  = ColumnTransformer(
    transformers = [
        ("inputer_pipe", inputer_pipe, num_var_1),
        ("numerical_pipe", median, num_var_2)
        ]
    )      
  
    
    pipe = make_pipeline(
        MedianByYTransformer(feature_cols=['insulina', 'glicose']),
        DecisionTreeDiscretiser(
            variables=['insulina'],
            regression=False, 
            scoring='accuracy',
            random_state=23),        
        Winsorizer(variables=num_var_2, capping_method='iqr'),
        preprocessor.set_output(transform="pandas")       
        )
    
    return pipe