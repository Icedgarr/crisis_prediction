from typing import List

from pandas import DataFrame

from crisis_prediction.preprocessors.base import AlphaPreprocessor


class AlphaFeaturesPreprocessor(AlphaPreprocessor):
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        return data
