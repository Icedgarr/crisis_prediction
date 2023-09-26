import datetime

import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import dummitize, first_known_to_date


class ContactEventFeatures(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), t_weeks_stats=[4, 8, 12, 16, 20, 24],
                 in_between_t_weeks_stats_to_keep=[4, 12]):
        super().__init__(end_date)
        self.event_code_categories = ['fso', 'gro', 'ts', 'carer', 'rev', 'other']
        self.event_code_columns = ['contact_event_code_{}'.format(name) for name in self.event_code_categories]
        self.t_weeks_stats = t_weeks_stats
        self.in_between_t_weeks_stats_to_keep = in_between_t_weeks_stats_to_keep
        self.columns_for_t_weeks_stats = {'contacts_sum': 'sum', 'contact_unplanned_sum': 'sum',
                                          'contact_dna_sum': 'sum'}

    def transform(self, data):
        """
        Takes a dictionary of dataframes and returns the features related to Contacts indexed by patient,
        year and week.
        """
        contacts_data = data['contacts_table']
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if contacts_data.empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        contacts_data.set_index(['anonymous_pat_id', 'contacts_datetime'], inplace=True)
        contacts_data['contact_dna'] = contacts_data['attendance'].map({
            'Did not attend (DNA) or not in': 1}).fillna(0)
        contacts_data['contact_unplanned'] = contacts_data['contact_service_code'].map({'Unplanned': 1, 'Planned': 0})
        contacts_data = self.add_event_code_category(contacts_data)
        contacts_features = self.add_event_stat_features(contacts_data, patients_first_known_date)
        contacts_features = self.add_time_since_last_features(contacts_features)
        contacts_features['contact_within_last_4_weeks'] = (
                contacts_features['time_since_last_contacts_max'] <= 4).astype(
            int)
        contacts_features['contact_within_last_24_weeks'] = (
                contacts_features['time_since_last_contacts_max'] <= 24).astype(
            int)
        last_contact_dna = (contacts_features['time_since_last_contact_dna_max'] ==
                            contacts_features['time_since_last_contacts_max'])
        contacts_features['contact_dna_without_followup'] = (
                last_contact_dna & ((contacts_features['contact_dna_sum'] == contacts_features['contacts_sum']) |
                                    (contacts_features['contacts_max'] == 0))
        ).astype(int)
        contacts_features = self.add_event_t_weeks_stat_features(contacts_features, self.columns_for_t_weeks_stats,
                                                                 self.t_weeks_stats)
        contacts_features = self.add_event_ever_stat_features(contacts_features, self.columns_for_t_weeks_stats)
        contacts_features = self.drop_intermediary_t_weeks_stats_columns(contacts_features,
                                                                         self.columns_for_t_weeks_stats,
                                                                         self.in_between_t_weeks_stats_to_keep)
        contacts_features['contact_risk_dna_indicator'] = (
                (1 - (contacts_features['contact_dna_sum_ever'] / contacts_features['contacts_sum_ever'])) *
                contacts_features[
                    'contact_dna_without_followup']).fillna(0)
        return contacts_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {column: np.nan if 'time_since_last' in column else 0 for column in
                             self._get_out_column_names()}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def add_event_code_category(self, contacts_data):
        contacts_data[self.event_code_columns] = dummitize(contacts_data['contact_event_code'].str.lower(),
                                                           self.event_code_columns)
        return contacts_data

    def add_event_stat_features(self, contacts_data, patients_first_known_date):
        stats_dict = {
            **{'contacts': ['sum', 'min', 'max'],
               'contact_unplanned': ['sum', 'max'],
               'contact_dna': ['sum', 'max']},
            **{k: 'max' for k in self.event_code_columns}
        }
        contacts_features = self.event_stats_per_week(
            contacts_data, patients_first_known_date,
            column_prefix='contacts',
            stats=stats_dict).sort_index()
        contacts_features.rename(columns={
            **{'{}_max'.format(k): k for k in self.event_code_columns}}, inplace=True)
        return contacts_features.astype(int)

    def add_event_t_weeks_stat_features(self, contacts_features,
                                        columns_for_t_weeks_stats,
                                        t=[4, 8, 12, 16, 20, 24]):
        contacts_features = self.event_stats_per_t_weeks(data=contacts_features,
                                                         columns_for_t_weeks_stats=columns_for_t_weeks_stats, t=t)
        return contacts_features

    def add_event_ever_stat_features(self, contacts_features,
                                     columns_for_t_weeks_stats):
        contacts_features = self.event_stats_ever(data=contacts_features,
                                                  columns_for_t_weeks_stats=columns_for_t_weeks_stats)
        return contacts_features

    def drop_intermediary_t_weeks_stats_columns(self, contacts_features, columns_for_t_weeks_stats,
                                                in_between_t_weeks_stats_to_keep):
        t_weeks_to_drop = [i for i in range(1, max(self.t_weeks_stats) + 1) if
                           i not in range(in_between_t_weeks_stats_to_keep[0], in_between_t_weeks_stats_to_keep[1])]
        intermediary_columns_to_drop = ['{}_{}_{}'.format(col, i, 'weeks_ago') for col in columns_for_t_weeks_stats for
                                        i in t_weeks_to_drop]
        contacts_features.drop(columns=intermediary_columns_to_drop, inplace=True)
        return contacts_features

    def add_time_since_last_features(self, contacts_features):
        for col in self.event_code_columns + ['contacts_max', 'contact_unplanned_max', 'contact_dna_max']:
            contacts_features['time_since_last_{}'.format(col)] = \
                self.get_time_since_last_event(contacts_features[col].astype(bool))
        return contacts_features

    def _get_out_column_names(self):
        basic_cols = ['contacts_sum', 'contacts_min', 'contacts_max',
                      'contact_unplanned_sum', 'contact_unplanned_max',
                      'contact_dna_sum', 'contact_dna_max',
                      'time_since_last_contacts_max',
                      'time_since_last_contact_dna_max',
                      'time_since_last_contact_unplanned_max',
                      'contact_within_last_4_weeks',
                      'contact_within_last_24_weeks',
                      'contact_dna_without_followup',
                      'contact_risk_dna_indicator'
                      ]
        time_since_last_columns = ['time_since_last_{}'.format(col) for col in self.event_code_columns]
        stats_t_weeks_cols = ['{}_{}_{}'.format(col, i, 'weeks_ago') for i in
                              range(self.in_between_t_weeks_stats_to_keep[0], self.in_between_t_weeks_stats_to_keep[1])
                              for col in self.columns_for_t_weeks_stats.keys()
                              ] + ['{}_ever'.format(col) for col in self.columns_for_t_weeks_stats.keys()
                                   ] + ['{}_in_last_{}_{}'.format(col, k, 'weeks') for k in self.t_weeks_stats for col
                                        in self.columns_for_t_weeks_stats.keys()]

        fields = basic_cols + self.event_code_columns + time_since_last_columns + stats_t_weeks_cols
        return fields

    @property
    def schema_out(self):
        return {k: int for k in self._get_out_column_names()}
