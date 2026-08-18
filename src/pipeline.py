from src.models.ltv_model import create_ltv_pipeline
from src.models.churn_model import create_churn_pipeline

class CustomerValuePipeline:

    def __init__(self, ltv_model, churn_model):

        self.ltv_model = ltv_model
        self.churn_model = churn_model

    def fit(self, X, y_ltv, y_churn):

        self.ltv_model.fit(X, y_ltv)
        self.churn_model.fit(X, y_churn)

        return self

    def predict(self, X):

        ltv_prediction = self.ltv_model.predict(X)
        churn_probability = self.churn_model.predict(X)

        return ltv_prediction, churn_probability

    