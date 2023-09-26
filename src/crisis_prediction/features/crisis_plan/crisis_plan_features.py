import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import first_known_to_date


class CrisisPlanEventFeatures(EventFeature):

    def transform(self, data):
        """
        Takes a dictionary of dataframes and returns the features related to CrisisPlan indexed by patient,
        year and week.
        """
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['crisis_plan_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        crisis_plan_data = data['crisis_plan_table']
        crisis_plan_data.set_index(['anonymous_pat_id', 'plan_updated_date'], inplace=True)
        crisis_plan_features = self.add_event_stat_features(crisis_plan_data, patients_first_known_date)
        crisis_plan_features = self.add_time_since_last_features(crisis_plan_features)
        crisis_plan_features['crisis_plan_up_to_date'] = \
            (crisis_plan_features['time_since_last_crisis_plan_update'] <= 52).astype(int)
        return crisis_plan_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {'crisis_plan_up_to_date': 0,
                             'time_since_last_crisis_plan_update': np.nan,
                             'crisis_plan_update': 0}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def add_event_stat_features(self, crisis_plan_data, patients_first_known_date):
        stats_dict = {'crisis_plan_update': 'max'}
        crisis_plan_features = self.event_stats_per_week(
            crisis_plan_data, patients_first_known_date,
            column_prefix='crisis_plan_update',
            stats=stats_dict).sort_index()
        return crisis_plan_features.astype(int)

    def add_time_since_last_features(self, crisis_plan_features):
        crisis_plan_features['time_since_last_crisis_plan_update'] = \
            self.get_time_since_last_event(crisis_plan_features['crisis_plan_update'].astype(bool))
        return crisis_plan_features

    @property
    def schema_out(self):
        schema = {'crisis_plan_up_to_date': int,
                  'time_since_last_crisis_plan_update': int,
                  'crisis_plan_update': int}
        return schema
