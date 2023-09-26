import datetime

import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import dummitize, first_known_to_date


class BedDayEventFeatures(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), t_weeks_stats=[4, 8, 12, 16, 20, 24, 52],
                 in_between_t_weeks_stats_to_keep=[4, 12]):
        super().__init__(end_date)
        self.activity_categories = ['acute_assessment', 'medium_secure', 'rehab', 'hdu', 'picu', 'continuing_care',
                                    'other']
        self.activity_columns = ['hospitalization_activity_{}'.format(name) for name in self.activity_categories]
        self.level_of_obs_columns = ['hospitalization_level_of_obs_{}'.format(name) for name in [1, 2, 3, 4]]
        self.t_weeks_stats = t_weeks_stats
        self.in_between_t_weeks_stats_to_keep = in_between_t_weeks_stats_to_keep
        self.columns_for_t_weeks_stats = {'hospitalization_sum': 'sum',
                                          **{'hospitalization_level_of_obs_{}_sum'.format(num): 'sum' for num in
                                             range(1, 5)},
                                          **{'hospitalization_activity_{}_sum'.format(cat): 'sum' for cat in
                                             self.activity_categories}
                                          }

    def transform(self, data):
        """
        Takes a dictionary of dataframes and returns the features related to Hospitalizations indexed by patient,
        year and week.
        """
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['hospitalization_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        hospitalization_data = data['hospitalization_table']
        hospitalization_data.set_index(['anonymous_pat_id', 'date_in_bed'], inplace=True)
        hospitalization_data['hospitalization_activity'] = hospitalization_data['activity_category'] \
            .map(data['bed_day_activity_category_code']['Category'].to_dict())
        hospitalization_data['hospitalization_level_of_obs'] = hospitalization_data['level_of_observation'] \
            .map({'LEVEL1': '1', 'LEVEL2': '2', 'LEVEL3': '3', 'LEVEL4': '4'}).fillna('1')
        hospitalization_data = self.add_activity_and_level_of_obs_category(hospitalization_data)
        hospitalization_features = self.add_event_stat_features(hospitalization_data, patients_first_known_date)
        hospitalization_features = self.add_time_since_last_features(hospitalization_features)
        hospitalization_features = self.add_event_ever_stat_features(hospitalization_features,
                                                                     self.columns_for_t_weeks_stats)
        '''hospitalization_features = self.add_event_t_weeks_stat_features(hospitalization_features,
                                                                        self.columns_for_t_weeks_stats,
                                                                        self.t_weeks_stats)
        hospitalization_features = self.drop_intermediary_t_weeks_stats_columns(hospitalization_features,
                                                                                self.columns_for_t_weeks_stats,
                                                                                self.in_between_t_weeks_stats_to_keep)'''
        return hospitalization_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {column: np.nan if 'time_since_last' in column else 0 for column in
                             self._get_out_column_names()}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def add_activity_and_level_of_obs_category(self, data):
        data[self.activity_columns] = dummitize(data['hospitalization_activity'].str.lower(),
                                                self.activity_columns)
        data[self.level_of_obs_columns] = dummitize(data['hospitalization_level_of_obs'].str.lower(),
                                                    self.level_of_obs_columns)
        return data

    def add_event_stat_features(self, data, patients_first_known_date):
        stats_dict = {
            **{'hospitalization': ['sum', 'max']},
            **{k: 'sum' for k in self.activity_columns},
            **{k: 'sum' for k in self.level_of_obs_columns}
        }
        hospitalization_features = self.event_stats_per_week(
            data, patients_first_known_date,
            column_prefix='hospitalization',
            stats=stats_dict).sort_index()
        return hospitalization_features.astype(int)

    def add_event_t_weeks_stat_features(self, hospitalization_features,
                                        columns_for_t_weeks_stats,
                                        t=[4, 8, 12, 16, 20, 24, 52]):
        hospitalization_features = self.event_stats_per_t_weeks(data=hospitalization_features,
                                                                columns_for_t_weeks_stats=columns_for_t_weeks_stats,
                                                                t=t)
        return hospitalization_features

    def add_event_ever_stat_features(self, hospitalization_features,
                                     columns_for_t_weeks_stats):
        hospitalization_features = self.event_stats_ever(data=hospitalization_features,
                                                         columns_for_t_weeks_stats=columns_for_t_weeks_stats)
        return hospitalization_features

    def drop_intermediary_t_weeks_stats_columns(self, hospitalization_features, columns_for_t_weeks_stats,
                                                in_between_t_weeks_stats_to_keep):
        t_weeks_to_drop = [i for i in range(1, max(self.t_weeks_stats) + 1) if
                           i not in range(in_between_t_weeks_stats_to_keep[0], in_between_t_weeks_stats_to_keep[1])]
        intermediary_columns_to_drop = ['{}_{}_{}'.format(col, i, 'weeks_ago') for col in columns_for_t_weeks_stats for
                                        i in t_weeks_to_drop]
        hospitalization_features.drop(columns=intermediary_columns_to_drop, inplace=True)
        return hospitalization_features

    def add_time_since_last_features(self, contacts_features):
        for col in self.activity_columns + self.level_of_obs_columns + ['hospitalization']:
            contacts_features['time_since_last_{}'.format(col)] = \
                self.get_time_since_last_event(contacts_features['{}_sum'.format(col)].astype(bool))
        return contacts_features

    def _get_out_column_names(self):
        basic_cols = ['hospitalization_sum', 'hospitalization_max',
                      'time_since_last_hospitalization']
        time_since_last_columns = ['time_since_last_{}'.format(col) for col in
                                   self.activity_columns + self.level_of_obs_columns]
        event_columns = ['{}_sum'.format(k) for k in self.activity_columns + self.level_of_obs_columns]
        ever_columns = ['{}_ever'.format(col) for col in self.columns_for_t_weeks_stats.keys()]
        return basic_cols + time_since_last_columns + event_columns + ever_columns

    @property
    def schema_out(self):
        basic_cols = ['hospitalization_sum', 'hospitalization_max',
                      'time_since_last_hospitalization']
        time_since_last_columns = ['time_since_last_{}'.format(col) for col in
                                   self.activity_columns + self.level_of_obs_columns]
        schema_basic = {k: int for k in basic_cols}
        schema_events = {'{}_sum'.format(k): int for k in self.activity_columns + self.level_of_obs_columns}
        schema_time_since_last_cols = {k: int for k in time_since_last_columns}
        schema_ever = {'{}_ever'.format(col): int for col in self.columns_for_t_weeks_stats.keys()}
        return {**schema_basic, **schema_events, **schema_time_since_last_cols, **schema_ever}
