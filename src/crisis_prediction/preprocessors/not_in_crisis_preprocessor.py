from pandas import DataFrame

from crisis_prediction.preprocessors.base import AlphaPreprocessor


class NotInCrisisPreprocessor(AlphaPreprocessor):
    def __init__(self, in_crisis_column: str = 'in_crisis_period_burst_1week'):
        self.in_crisis_column = in_crisis_column

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        train_data = self.not_in_crisis_preprocessor(train_data)
        test_data = self.not_in_crisis_preprocessor(test_data)
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        return self.not_in_crisis_preprocessor(data)

    def not_in_crisis_preprocessor(self, data: DataFrame) -> DataFrame:
        data = data.loc[data[self.in_crisis_column] == 0]
        return data
