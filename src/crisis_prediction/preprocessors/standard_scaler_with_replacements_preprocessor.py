from typing import List

from pandas import DataFrame

from crisis_prediction.preprocessors.replace_99_to_nan_preprocessor import Replace99ToNanPreprocessor
from crisis_prediction.preprocessors.standard_scaler_preprocessor import StandardScalerPreprocessor
from crisis_prediction.preprocessors.replace_nan_to_0_preprocessor import ReplaceNanTo0Preprocessor


class StandardScalerWithReplacementsPreprocessor(Replace99ToNanPreprocessor, StandardScalerPreprocessor,
                                                 ReplaceNanTo0Preprocessor):
    def __init__(self, feature_names: List[str]):
        super().__init__(feature_names=feature_names)

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        train_data, test_data = super().preprocess(train_data=train_data, test_data=test_data)
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        data = super().preprocess_single_df(data=data)
        return data
