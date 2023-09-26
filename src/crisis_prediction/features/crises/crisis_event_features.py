import datetime

import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import dummitize, convert_camel_case_column_to_snake_case, first_known_to_date


class CrisisEventFeatures(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), t_weeks_stats=[4, 8, 12, 16, 20, 24],
                 in_between_t_weeks_stats_to_keep=[4, 12]):
        super().__init__(end_date)
        self.type_codes = ['TR', 'BM', 'IP', 'OOA']
        self.contact_methods = ['Contact', 'BM_BedMngmnt_Day', 'IP_BedDay', 'ST',
                                'BM_PDU_Day', 'BM_PoS_Day', 'OOA', 'RNC']
        self.contact_columns = ['crisis_contact_allocation_{}'.format(c) for c in self.contact_methods]
        self.type_columns = ['crisis_type_{}'.format(c) for c in self.type_codes]
        self.t_weeks_stats = t_weeks_stats
        self.in_between_t_weeks_stats_to_keep = in_between_t_weeks_stats_to_keep
        self.columns_for_t_weeks_stats = {'crisis_sum': 'sum', 'severity_max': 'max'}

    def transform(self, data):
        """This function takes a dictionary of dataframes
        with the table names as keys and returns features
        for crisis.
        """
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['crisis_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        crisis_data = data['crisis_table'].set_index(['anonymous_pat_id', 'event_date'])
        crisis_data = self.add_dummy_columns(crisis_data)
        crisis_data['severity'] = self.get_crisis_severity(crisis_data, data['crisis_severity'])
        crisis_features = self.add_event_stat_features(crisis_data, patients_first_known_date)
        crisis_features['time_since_last_crisis'] = self.get_time_since_last_event(
            crisis_features['crisis_max'].astype(bool))
        crisis_features.columns = [convert_camel_case_column_to_snake_case(col) for col in crisis_features.columns]
        crisis_features = self.add_event_t_weeks_stat_features(crisis_features, self.columns_for_t_weeks_stats,
                                                               self.t_weeks_stats)
        crisis_features = self.add_event_ever_stat_features(crisis_features, self.columns_for_t_weeks_stats)
        crisis_features = self.drop_intermediary_t_weeks_stats_columns(crisis_features, self.columns_for_t_weeks_stats,
                                                                       self.in_between_t_weeks_stats_to_keep)
        crisis_features['crisis_within_last_4_weeks'] = (crisis_features['time_since_last_crisis'] <= 4).astype(int)
        crisis_features['crisis_within_last_8_weeks'] = (crisis_features['time_since_last_crisis'] <= 8).astype(int)
        return crisis_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {column: np.nan if 'time_since_last' in column else 0 for column in
                             self._get_out_column_names()}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    @staticmethod
    def get_crisis_severity(crisis_data, crisis_severity):
        return crisis_data['crisis_contact_allocation'].map(crisis_severity['Severity'].to_dict())

    def add_dummy_columns(self, crisis_data):
        crisis_data[self.type_columns] = dummitize(
            crisis_data['crisis_type'], self.type_columns)
        crisis_data[self.contact_columns] = dummitize(
            crisis_data['crisis_contact_allocation'], self.contact_columns)
        return crisis_data

    def add_event_stat_features(self, crisis_data, patients_first_known_date):
        stats_dict = {
            **{'crisis': ['sum', 'min', 'max'],
               'severity': 'max'},
            **{k: 'max' for k in self.type_columns + self.contact_columns}
        }
        crisis_features = self.event_stats_per_week(
            crisis_data, patients_first_known_date,
            column_prefix='crisis',
            stats=stats_dict).sort_index()
        return crisis_features

    def add_event_t_weeks_stat_features(self, crisis_features, columns_for_t_weeks_stats, t=[4, 8, 12, 16, 20, 24]):
        crisis_features = self.event_stats_per_t_weeks(data=crisis_features,
                                                       columns_for_t_weeks_stats=columns_for_t_weeks_stats, t=t)
        return crisis_features

    def add_event_ever_stat_features(self, crisis_features, columns_for_t_weeks_stats):
        crisis_features = self.event_stats_ever(data=crisis_features,
                                                columns_for_t_weeks_stats=columns_for_t_weeks_stats)
        return crisis_features

    def drop_intermediary_t_weeks_stats_columns(self, crisis_features, columns_for_t_weeks_stats,
                                                in_between_t_weeks_stats_to_keep):
        t_weeks_to_drop = [i for i in range(1, max(self.t_weeks_stats) + 1) if
                           i not in range(in_between_t_weeks_stats_to_keep[0], in_between_t_weeks_stats_to_keep[1])]
        intermediary_columns_to_drop = ['{}_{}_{}'.format(col, i, 'weeks_ago') for col in
                                        columns_for_t_weeks_stats.keys() for i in t_weeks_to_drop]
        crisis_features.drop(columns=intermediary_columns_to_drop, inplace=True)
        return crisis_features

    def _get_out_column_names(self):
        basic_cols = ['crisis_max', 'crisis_min', 'crisis_sum', 'severity_max', 'time_since_last_crisis',
                      'crisis_within_last_4_weeks', 'crisis_within_last_8_weeks']
        stats_t_weeks_cols = ['{}_{}_{}'.format(col, i, 'weeks_ago') for i in
                              range(self.in_between_t_weeks_stats_to_keep[0], self.in_between_t_weeks_stats_to_keep[1])
                              for col in self.columns_for_t_weeks_stats.keys()
                              ] + ['{}_ever'.format(col) for col in self.columns_for_t_weeks_stats.keys()
                                   ] + ['{}_in_last_{}_{}'.format(col, k, 'weeks') for k in self.t_weeks_stats for col
                                        in self.columns_for_t_weeks_stats.keys()
                                        ]

        schema_types = [convert_camel_case_column_to_snake_case(k) + '_max' for k in self.type_columns]
        schema_contact = [convert_camel_case_column_to_snake_case(k) + '_max' for k in self.contact_columns]

        fields = schema_types + schema_contact + basic_cols + stats_t_weeks_cols
        return fields

    @property
    def schema_out(self):
        return {k: int for k in self._get_out_column_names()}
