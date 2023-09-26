import numpy as np

from crisis_prediction.features.last_crisis.last_crisis_features import LastCrisisFeatures


class LastCrisisBedDaysFeatures(LastCrisisFeatures):

    def transform(self, data):
        if data['BedDaysDuringCrisisPeriod'].empty:
            return self.transform_empty(data['InCrisisPeriod'].copy())
        columns_to_drop = ['start_crisis_period_monday', 'end_crisis_period_monday']
        last_crisis_features = self.add_during_crisis_features(data['BedDaysDuringCrisisPeriod'].copy(),
                                                               data['InCrisisPeriod'].copy(), columns_to_drop)
        return last_crisis_features

    def transform_empty(self, in_crisis_df):
        columns = ['number_of_leave_days_last_crisis', 'level_of_obs_max_last_crisis', 'number_of_bed_days_last_crisis',
                   'max_length_stay_last_crisis']
        columns += ['level_of_obs_{}_last_crisis'.format(num) for num in range(1, 5)]
        columns += ['level_of_obs_{}_number_of_days_last_crisis'.format(num) for num in range(1, 5)]
        in_crisis_col = 'in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst)

        nan_condition = ((in_crisis_df[in_crisis_col].shift(1) == 1) &
                         (in_crisis_df[in_crisis_col] == 0)).astype(int).cumsum() < 1
        for col in columns:
            in_crisis_df[col] = 0
        in_crisis_df.loc[nan_condition, columns] = np.nan
        return in_crisis_df

    @property
    def schema_out(self):
        schema = {
            **{'level_of_obs_{}_last_crisis'.format(num): int for num in range(1, 5)},
            **{'level_of_obs_{}_number_of_days_last_crisis'.format(num): int for num in range(1, 5)},
            **{'number_of_leave_days_last_crisis': int,
               'level_of_obs_max_last_crisis': int,
               'number_of_bed_days_last_crisis': int,
               'max_length_stay_last_crisis': int,
               }
        }
        return schema
