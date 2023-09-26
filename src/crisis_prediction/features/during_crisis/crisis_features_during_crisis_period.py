import datetime

import pandas as pd
from isoweek import Week

from crisis_prediction.features.crisis_period_feature import CrisisPeriodFeature


class CrisisFeaturesDuringCrisisPeriod(CrisisPeriodFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), weeks_before_new_burst=1):
        super().__init__(end_date=end_date)
        self.weeks_before_new_burst = weeks_before_new_burst

    def transform(self, data):
        if data['crisis_table'].empty:
            return pd.DataFrame()
        crisis_burst, crisis_data = data['InCrisisPeriod'], data['crisis_table'].set_index(
            ['anonymous_pat_id', 'year', 'week'])
        crisis_data['severity'] = self.get_crisis_severity(crisis_data, data['crisis_severity'])
        crisis_periods = self.get_crisis_periods(crisis_data, crisis_burst)
        crisis_periods = self.get_aggregations_features(crisis_periods)
        crisis_periods = self.rename_feature_columns(crisis_periods)
        return crisis_periods

    @staticmethod
    def get_crisis_severity(crisis_data, crisis_severity):
        return crisis_data['crisis_contact_allocation'].map(crisis_severity['Severity'].to_dict())

    def get_crisis_periods(self, crisis_data, crisis_burst):
        crisis_periods = crisis_burst.join(crisis_data, how='left').sort_index()
        crisis_periods = crisis_periods.loc[
            crisis_periods['in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst)] != 0]
        return crisis_periods

    def get_aggregations_features(self, crisis_periods):
        crisis_periods['monday_of_week'] = crisis_periods.apply(
            lambda row: Week(int(row.name[1]), int(row.name[2])).day(0), axis=1)
        crisis_periods = crisis_periods.groupby(
            ['anonymous_pat_id', 'number_crisis_burst_{}week'.format(self.weeks_before_new_burst)]).agg(
            {'event_date': ['count', 'nunique'], 'severity': 'max', 'monday_of_week': ['first', 'last']})
        return crisis_periods

    @staticmethod
    def rename_feature_columns(crisis_periods):
        rename_columns = {'monday_of_week_first': 'start_crisis_period_monday',
                          'monday_of_week_last': 'end_crisis_period_monday',
                          'severity_max': 'max_severity_crisis', 'event_date_count': 'number_of_crisis',
                          'event_date_nunique': 'number_of_days_in_crisis'}
        crisis_periods.columns = crisis_periods.columns.map('_'.join)
        crisis_periods.rename(columns=rename_columns, inplace=1)
        return crisis_periods

    @property
    def schema_out(self):
        return {}
