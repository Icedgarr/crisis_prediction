from typing import List

import numpy as np
from pandas import DataFrame

from crisis_prediction.preprocessors.base_preprocess_features import AlphaFeaturesPreprocessor


class Replace99ToNanPreprocessor(AlphaFeaturesPreprocessor):
    def __init__(self, feature_names: List[str], **kw):
        self.feature_names = feature_names
        super().__init__(feature_names)

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        train_data = self.replace_99_to_nan(train_data)
        test_data = self.replace_99_to_nan(test_data)
        train_data, test_data = super().preprocess(train_data=train_data, test_data=test_data)
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        data = self.replace_99_to_nan(data)
        data = super().preprocess_single_df(data=data)
        return data

    def replace_99_to_nan(self, data):
        data[self.feature_names] = data[self.feature_names].replace(-99, np.nan)
        return data
