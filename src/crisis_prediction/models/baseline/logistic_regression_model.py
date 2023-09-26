from typing import List, Dict

from numpy import array
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from crisis_prediction.models.base import AlphaModel

LOGISTIC_MODEL_HYPERPARAMETERS = dict(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='liblinear',
    max_iter=100,
    multi_class='ovr',
    verbose=0,
    warm_start=False,
    n_jobs=1
)


class LogisticRegressionModel(AlphaModel):
    def __init__(self, feature_names: List[str], target_name: str, hyperparameters: Dict = None,
                 fit_params: Dict = None):
        super().__init__(feature_names=feature_names, target_name=target_name, fit_params=fit_params)
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = dict(LOGISTIC_MODEL_HYPERPARAMETERS, **hyperparameters)
        self.model = LogisticRegression(**self.hyperparameters)

    def fit(self, data: DataFrame):
        self.model.fit(data[self.feature_names], data[self.target_name], **self.fit_params)

    def predict(self, data: DataFrame) -> array:
        return self.model.predict_proba(data[self.feature_names])[:, 1]
