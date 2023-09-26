from pandas import DataFrame

from crisis_prediction.preprocessors.base import AlphaPreprocessor


class AfterCriteriaPreprocessor(AlphaPreprocessor):

    def __init__(self, minimum_num_crisis=10, minimum_num_crisis_burst=1, minimum_num_months=3, remove_dead=True,
                 crisis_burst_column='number_crises_burst_1week_passed'):
        self.minimum_num_crisis = minimum_num_crisis
        self.minimum_num_months = minimum_num_months
        self.minimum_num_crisis_burst = minimum_num_crisis_burst
        self.remove_dead = remove_dead
        self.crisis_burst_column = crisis_burst_column

    def preprocess(self, train_data: DataFrame, test_data: DataFrame) -> (DataFrame, DataFrame):
        train_data = self.after_criteria(train_data)
        test_data = self.after_criteria(test_data)
        return train_data, test_data

    def preprocess_single_df(self, data: DataFrame) -> DataFrame:
        return self.after_criteria(data)

    def after_criteria(self, data):
        cond = ((data['months_since_known'] >= self.minimum_num_months) &
                (data['crisis_sum_ever'] >= self.minimum_num_crisis) &
                (data[self.crisis_burst_column] >= self.minimum_num_crisis_burst))
        if self.remove_dead:
            cond = cond & self.remove_dead_condition(data)
        return data[cond]

    def remove_dead_condition(self, data):
        death_nan = ((data['static_death_year'].isnull()) | (data['static_death_year'] == -99))
        death_after_year = data['static_death_year'] > data['year']
        death_in_year_after_week = ((data['static_death_year'] == data['year']) &
                                    (data['static_death_month'] > data['week'] / 4))
        return death_nan | death_after_year | death_in_year_after_week
