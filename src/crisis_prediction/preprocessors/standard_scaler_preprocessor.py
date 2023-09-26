from typing import List

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from crisis_prediction.preprocessors.base_preprocess_features import AlphaFeaturesPreprocessor


class StandardScalerPreprocessor(AlphaFeaturesPreprocessor):
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        super().__init__(feature_names)

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        self.fit_scaler(train_data)
        train_data[self.feature_names] = self.transform_data(train_data)
        test_data[self.feature_names] = self.transform_data(test_data)
        train_data, test_data = super().preprocess(train_data=train_data, test_data=test_data)
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        self.fit_scaler(data)
        data[self.feature_names] = self.transform_data(data)
        data = super().preprocess_single_df(data=data)
        return data

    def fit_scaler(self, data):
        self.scaler.fit(data[self.feature_names])

    def transform_data(self, data):
        return self.scaler.transform(data[self.feature_names])
