import datetime

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import isocalendar_week, isocalendar_year


class LastCrisisFeatures(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), weeks_before_new_burst=1):
        super().__init__(end_date=end_date)
        self.weeks_before_new_burst = weeks_before_new_burst

    def transform(self, data):
        return NotImplementedError

    @staticmethod
    def add_during_crisis_features(during_crisis_period, last_crisis_period, columns_to_drop):
        during_crisis_period['year'] = during_crisis_period['end_crisis_period_monday'].apply(isocalendar_year)
        during_crisis_period['week'] = during_crisis_period['end_crisis_period_monday'].apply(isocalendar_week)
        during_crisis_period = during_crisis_period.reset_index().set_index(['anonymous_pat_id', 'year', 'week'])
        last_crisis_features = last_crisis_period.join(during_crisis_period)
        new_columns = [col for col in last_crisis_features.columns if
                       col not in columns_to_drop and '_last_crisis' not in col]
        last_crisis_features.loc[:, new_columns] = last_crisis_features.loc[:, new_columns].shift(1).ffill()
        last_crisis_features.rename(columns={col: col + '_last_crisis' for col in new_columns}, inplace=True)
        feature_columns = [col for col in last_crisis_features.columns if '_last_crisis' in col]
        return last_crisis_features[feature_columns]

    def schema_out(self):
        return NotImplementedError
