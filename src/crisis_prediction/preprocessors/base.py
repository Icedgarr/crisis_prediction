from abc import ABCMeta, abstractmethod

from pandas import DataFrame


class AlphaPreprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        pass

    @abstractmethod
    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        pass
