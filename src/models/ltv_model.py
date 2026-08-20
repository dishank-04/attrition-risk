import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from src.preprocessing import preprocessor


def create_ltv_pipeline(model, transform_target=True):


    if transform_target:

        target_ltv_transformation = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

        ltv_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', target_ltv_transformation)])

    else:
        ltv_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model)])

    return ltv_pipeline


