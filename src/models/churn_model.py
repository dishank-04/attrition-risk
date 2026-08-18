import numpy as np

from sklearn.pipeline import Pipeline
from src.preprocessing import preprocessor

def create_churn_pipeline(model):

    churn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', model)])

    return churn_pipeline
