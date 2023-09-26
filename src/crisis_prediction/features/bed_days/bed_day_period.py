import numpy as np
import pandas as pd

from crisis_prediction.features.base import Feature
from crisis_prediction.features.utils import dummitize


class BedDayPeriod(Feature):
    level_of_obs_columns = ['level_of_obs_1', 'level_of_obs_2', 'level_of_obs_3', 'level_of_obs_4']

    def transform(self, data):
        if data['hospitalization_table'].empty:
            return pd.DataFrame()
        bed_days_data = data['hospitalization_table'].copy().sort_values(['anonymous_pat_id', 'date_in_bed'])
        bed_days_data.rename(columns={'level_of_observation': 'level_of_obs',
                                      'date_leave': 'leave_day',
                                      'date_in_bed': 'bed_day'}, inplace=True)
        bed_days_data['level_of_obs'] = bed_days_data['level_of_obs'] \
            .map({'LEVEL1': 1, 'LEVEL2': 2, 'LEVEL3': 3, 'LEVEL4': 4}).fillna(1).astype(int)
        bed_days_data[self.level_of_obs_columns] = dummitize(bed_days_data['level_of_obs'], self.level_of_obs_columns)
        bed_days_data['bed_day_period_number'] = self.compute_bed_day_period_number(bed_days_data['date_admission'])
        bed_days_features = self.compute_bed_day_features(bed_days_data).reset_index() \
            .drop(columns='bed_day_period_number')

        return bed_days_features.set_index(['anonymous_pat_id', 'start_bed_day_period', 'end_bed_day_period'])

    def compute_bed_day_features(self, bed_days_data):
        bed_days_features = bed_days_data.groupby(['anonymous_pat_id', 'bed_day_period_number']).agg({
            **{col: ['max', 'sum'] for col in self.level_of_obs_columns},
            **{'level_of_obs': 'max',
               'leave_day': 'sum',
               'bed_day': ['count', 'max', 'min']}
        })
        bed_days_features.columns = bed_days_features.columns.map('_'.join)
        bed_days_features.rename(columns={
            **{'level_of_obs_{}_max'.format(num): 'level_of_obs_{}'.format(num) for num in range(1, 5)},
            **{'level_of_obs_{}_sum'.format(num): 'level_of_obs_{}_number_of_days'.format(num) for num in range(1, 5)},
            **{'bed_day_count': 'number_of_bed_days', 'leave_day_sum': 'number_of_leave_days',
               'bed_day_min': 'start_bed_day_period', 'bed_day_max': 'end_bed_day_period'}
        }, inplace=True)
        return bed_days_features

    @staticmethod
    def compute_bed_day_period_number(admit_data):
        admit_days = admit_data[admit_data == 1]
        bed_day_period_number = admit_data.map({0: np.nan})
        bed_day_period_number[admit_days.index] = range(1, len(admit_days) + 1)
        return bed_day_period_number.ffill()

    @property
    def schema_out(self):
        schema = {}

        return schema
