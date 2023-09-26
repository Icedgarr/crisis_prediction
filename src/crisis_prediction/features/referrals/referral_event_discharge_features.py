import datetime

import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import dummitize, convert_camel_case_column_to_snake_case, first_known_to_date


class ReferralDischargeEventFeatures(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), t_weeks_stats=[4, 8, 12, 16, 20, 24],
                 in_between_t_weeks_stats_to_keep=[4, 12]):
        super().__init__(end_date=end_date)
        self.discharge_categories = ['Complete', 'DNA', 'Declined', 'Internal',
                                     'NoMH', 'Not Suitable', 'Other', 'Security']
        self.discharge_columns = ['referral_event_discharge_category_{}'.format(c)
                                  for c in self.discharge_categories]
        self.t_weeks_stats = t_weeks_stats
        self.in_between_t_weeks_stats_to_keep = in_between_t_weeks_stats_to_keep
        self.columns_for_t_weeks_stats = {'referral_discharge_sum': 'sum'}

    def transform(self, data):
        """This function takes a dictionary of dataframes
        with the table names as keys and returns features
        for referrals.
        """
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['referral_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        referral_data = data['referral_table'].set_index(['anonymous_pat_id', 'discharge_date'])
        referral_data[self.discharge_columns] = dummitize(referral_data['discharge_category']
                                                          .rename('referral_event_discharge_category'),
                                                          self.discharge_columns)
        referral_features = self.add_event_stats(referral_data, patients_first_known_date)
        referral_features.columns = [convert_camel_case_column_to_snake_case(c).replace('_max', '')
                                     for c in referral_features.columns]
        referral_features = self.add_time_since_last_features(referral_features)
        referral_features = self.add_event_t_weeks_stat_features(referral_features, self.columns_for_t_weeks_stats,
                                                                 self.t_weeks_stats)
        referral_features = self.add_event_ever_stat_features(referral_features, self.columns_for_t_weeks_stats)
        referral_features = self.drop_intermediary_t_weeks_stats_columns(referral_features,
                                                                         self.columns_for_t_weeks_stats,
                                                                         self.in_between_t_weeks_stats_to_keep)
        referral_features['referral_discharge_within_last_4_weeks'] = (
                referral_features['time_since_last_referral_discharge'] <= 4).astype(
            int)
        referral_features['referral_discharge_within_last_8_weeks'] = (
                referral_features['time_since_last_referral_discharge'] <= 8).astype(
            int)
        return referral_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {k: np.nan if 'time_since_last' in k else 0 for k in self.get_out_column_names()}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def add_event_stats(self, referral_data, patients_first_known_date):
        stats_dict = {
            **{'referral_discharge': ['sum', 'max']},
            **{c: 'max' for c in self.discharge_columns}
        }
        referral_data = self.event_stats_per_week(
            referral_data[self.discharge_columns], patients_first_known_date,
            column_prefix='referral_discharge',
            stats=stats_dict)
        return referral_data

    def add_event_t_weeks_stat_features(self, referral_features, columns_for_t_weeks_stats, t=[4, 8, 12, 16, 20, 24]):
        referral_features = self.event_stats_per_t_weeks(data=referral_features,
                                                         columns_for_t_weeks_stats=columns_for_t_weeks_stats, t=t)
        return referral_features

    def add_event_ever_stat_features(self, referral_features, columns_for_t_weeks_stats):
        referral_features = self.event_stats_ever(data=referral_features,
                                                  columns_for_t_weeks_stats=columns_for_t_weeks_stats)
        return referral_features

    def drop_intermediary_t_weeks_stats_columns(self, referral_features, columns_for_t_weeks_stats,
                                                in_between_t_weeks_stats_to_keep):
        t_weeks_to_drop = [i for i in range(1, max(self.t_weeks_stats) + 1) if
                           i not in range(in_between_t_weeks_stats_to_keep[0], in_between_t_weeks_stats_to_keep[1])]
        intermediary_columns_to_drop = ['{}_{}_{}'.format(col, i, 'weeks_ago') for col in
                                        columns_for_t_weeks_stats.keys() for i in t_weeks_to_drop]
        referral_features.drop(columns=intermediary_columns_to_drop, inplace=True)
        return referral_features

    def add_time_since_last_features(self, referral_features):
        for col in self.discharge_columns + ['referral_discharge']:
            referral_features['time_since_last_{}'.format(convert_camel_case_column_to_snake_case(col))] = \
                self.get_time_since_last_event(
                    referral_features[convert_camel_case_column_to_snake_case(col)].astype(bool))
        return referral_features

    def get_out_column_names(self):
        discharge_source_columns = [convert_camel_case_column_to_snake_case(c)
                                    for c in self.discharge_columns]
        stats_t_weeks_cols = ['{}_{}_{}'.format(col, i, 'weeks_ago') for i in
                              range(self.in_between_t_weeks_stats_to_keep[0],
                                    self.in_between_t_weeks_stats_to_keep[1])
                              for col in self.columns_for_t_weeks_stats.keys()
                              ] + ['{}_ever'.format(col) for col in
                                   self.columns_for_t_weeks_stats.keys()
                                   ] + ['{}_in_last_{}_{}'.format(col, k, 'weeks') for k in self.t_weeks_stats for col
                                        in self.columns_for_t_weeks_stats.keys()
                                        ]

        time_since_last_columns = ['time_since_last_{}'.format(convert_camel_case_column_to_snake_case(c))
                                   for c in self.discharge_columns]
        fields = discharge_source_columns + time_since_last_columns + [
            'referral_discharge_sum', 'referral_discharge',
            'time_since_last_referral_discharge',
            'referral_discharge_within_last_4_weeks',
            'referral_discharge_within_last_8_weeks'] + stats_t_weeks_cols
        return fields

    @property
    def schema_out(self):

        schema = {k: int for k in self.get_out_column_names()}
        return schema
