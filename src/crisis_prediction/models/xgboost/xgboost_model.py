from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from xgboost import XGBClassifier

from crisis_prediction.models.base import AlphaModel

default = {'colsample_bytree': 0.41536014182042663,
           'gamma': 249.99090642487218,
           'learning_rate': 0.009390673710943508,
           'max_depth': 12.0,
           'min_child_weight': 113.0,
           'n_estimators': 1003.0,
           'reg_alpha': 19.339000391672123,
           'reg_lambda': 9.687085346278936,
           'scale_pos_weight': 0.5000522414460131,
           'subsample': 0.7846115074512965,
           'n_jobs': 60
           }


class XGBoostModel(AlphaModel):
    def __init__(self, feature_names: List[str], target_name: str, hyperparameters=default, fit_params={}):
        super().__init__(feature_names=feature_names, target_name=target_name)
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = dict(default, **hyperparameters)
        for param in ['n_jobs', 'min_child_weight', 'max_depth', 'n_estimators']:
            try:
                hyperparameters[param] = int(hyperparameters[param])
            except KeyError:
                continue
        self.scale_pos_weight = hyperparameters['scale_pos_weight']
        self.Model = XGBClassifier
        self.model = None
        self.fit_params = fit_params

    def _set_scale_pos_weight(self, target):
        """
            Scale pos weight is calculated based on the predefined weight and number of zeros and ones in the target,
            as follows: scale_pos_weight * (num_zeros / num_ones),
            where num_zeros and num_ones would be the number of the target values that are 0
            and the number of target values that are 1 in the historical data.
        """
        prop_zeros = (np.sum(1 - target.values) / np.sum(target.values))
        self.hyperparameters['scale_pos_weight'] = float(self.scale_pos_weight * prop_zeros)

    def fit(self, data: DataFrame):
        self._set_scale_pos_weight(data[self.target_name])
        self.model = self.Model(**self.hyperparameters)
        self.model.fit(data[self.feature_names], data[self.target_name], **self.fit_params)

    def predict(self, data: DataFrame):
        predictions = self.model.predict_proba(data[self.feature_names])[:, 1]
        return predictions

    def feature_importance(self):
        feature_importances = pd.DataFrame(index=self.feature_names)
        for imp_type in ['gain', 'weight', 'cover']:
            feature_importances_type = pd.Series(
                self.model.get_booster().get_score(importance_type=imp_type))
            feature_importances_type = feature_importances_type / feature_importances_type.sum()
            feature_importances_type.name = imp_type
            feature_importances = feature_importances.join(feature_importances_type, how='outer')
        feature_importances.sort_values('gain', ascending=False, inplace=True)
        feature_importances.index.name = 'feature'
        feature_importances.reset_index(inplace=True)
        return feature_importances
