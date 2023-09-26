import pandas as pd

from crisis_prediction.features.crisis_period_feature import CrisisPeriodFeature


class BedDaysDuringCrisisPeriod(CrisisPeriodFeature):

    def transform(self, data):
        crisis_periods = data['CrisisFeaturesDuringCrisisPeriod']
        if data['BedDayPeriod'].empty:
            return self.transform_empty(crisis_periods)
        bed_day_periods = data['BedDayPeriod']
        bed_day_features = crisis_periods.apply(self.get_aggregation_features(bed_day_periods.reset_index()), axis=1)
        return bed_day_features

    def transform_empty(self, crisis_periods):
        columns = ['number_of_bed_days', 'number_of_leave_days', 'level_of_obs_max', 'max_length_stay']
        columns += ['level_of_obs_{}'.format(num) for num in range(1, 5)]
        columns += ['level_of_obs_{}_number_of_days'.format(num) for num in range(1, 5)]
        for col in columns:
            crisis_periods[col] = 0
        return crisis_periods

    @staticmethod
    def get_aggregation_features(bed_day_periods):
        def currying_function(crisis_period):
            start = pd.to_datetime(crisis_period['start_crisis_period_monday'])
            end = pd.to_datetime(crisis_period['end_crisis_period_monday'])
            bed_day_periods_in_crisis_period = bed_day_periods[(bed_day_periods['start_bed_day_period'] >= start) &
                                                               (bed_day_periods['end_bed_day_period'] <= end)]
            features = {
                **{'level_of_obs_{}'.format(num): max(bed_day_periods_in_crisis_period['level_of_obs_{}'.format(num)],
                                                      default=0) for num in range(1, 5)},
                **{
                    'level_of_obs_{}_number_of_days'.format(num):
                        sum(bed_day_periods_in_crisis_period['level_of_obs_{}_number_of_days'.format(num)])
                    for num in range(1, 5)},
                **{'number_of_bed_days': sum(bed_day_periods_in_crisis_period['number_of_bed_days']),
                   'max_length_stay': max(bed_day_periods_in_crisis_period['number_of_bed_days'], default=0),
                   'number_of_leave_days': sum(bed_day_periods_in_crisis_period['number_of_leave_days']),
                   'level_of_obs_max': max(bed_day_periods_in_crisis_period['level_of_obs_max'], default=0),
                   'start_crisis_period_monday': start,
                   'end_crisis_period_monday': end}
            }
            return pd.Series(features)

        return currying_function

    @property
    def schema_out(self):
        schema = {}
        return schema
