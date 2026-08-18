import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline

'''
Step-1)

Frequency -> log1p
Monetary -> log1p
AvgBasketSize -> log1p
Recency -> unchanged
Tenure -> unchanged

Step-2) 

Then Standard Scaling applied to all the Features to keep the scale same for all features.
'''

LOG_FEATURES = ["Frequency", "Monetary", "AvgBasketSize"]
NUMERIC_FEATURES = ["Recency", "Tenure"]

log_tranformer = FunctionTransformer(np.log1p)
log_scaler = StandardScaler()
numeric_scaler = StandardScaler()


log_features_pipeline = Pipeline(steps=[('transformer', log_tranformer),
                               ('scaler', log_scaler)])


numeric_features_pipeline = Pipeline(steps=[('scaler', numeric_scaler)])


preprocessor = ColumnTransformer(transformers=[('log_features', log_features_pipeline, LOG_FEATURES),
                                               ('numeric_features', numeric_features_pipeline, NUMERIC_FEATURES)],
                                remainder="drop",
                                verbose_feature_names_out=False)










