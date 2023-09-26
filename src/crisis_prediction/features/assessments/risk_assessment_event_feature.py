import numpy as np

from crisis_prediction.features.event_feature import EventFeature
from crisis_prediction.features.utils import convert_camel_case_column_to_snake_case, add_missing_columns, \
    first_known_to_date


class RiskAssessmentEventFeature(EventFeature):
    risk_columns = ['risk_suicide', 'risk_substance_misuse', 'risk_self_neglect',
                    'risk_forensic_care', 'risk_self_harm', 'risk_to_children',
                    'risk_of_absconding', 'risk_med_phys', 'risk_of_violence',
                    'risk_of_accident', 'risk_of_harm_from_others', 'risk_assessment']

    def transform(self, data):
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['risk_screening_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        risk_data = data['risk_screening_table'].copy().set_index(['anonymous_pat_id', 'screening_datetime'])
        risk_data.columns = [self._to_correct_column_format(col) for col in risk_data.columns]
        risk_data.replace({'Y': 1, 'N': 0, 'DN': 1}, inplace=True)
        risk_data = add_missing_columns(risk_data, self.risk_columns[:-1]).fillna(0)
        risk_features = self.add_event_stat_features(risk_data, patients_first_known_date).astype(int)
        risk_features.rename(
            columns={'risk': 'risk_assessment'},
            inplace=True)
        risk_features = self.add_time_since_last_features(risk_features)

        risk_features['risk_assessment_not_up_to_date'] = (
            ~(risk_features['time_since_last_risk_assessment'] < 52)).astype(int)
        risk_features.loc[risk_features['risk_assessment_not_up_to_date'] == 1, self.risk_columns] = np.nan
        risk_features.loc[risk_features['risk_assessment_not_up_to_date'] == 1, 'risk_assessment'] = 0
        return risk_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {**{'risk_assessment': 0},
                             **{column: np.nan for column in self.risk_columns if column != 'risk_assessment'},
                             **{'time_since_last_{}'.format(column): np.nan for column in self.risk_columns},
                             **{'risk_assessment_not_up_to_date': 1}}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def add_event_stat_features(self, risk_data, patients_first_known_date):
        stats_dict = {
            **{k: 'max' for k in self.risk_columns[:-1] + ['risk']}
        }
        risk_features = self.event_stats_per_week(
            risk_data, patients_first_known_date,
            column_prefix='risk',
            stats=stats_dict).sort_index()
        return risk_features

    def add_time_since_last_features(self, risk_features):
        for col in self.risk_columns:
            risk_features['time_since_last_{}'.format(col)] = self.get_time_since_last_event(
                risk_features[col].astype(bool))
        return risk_features

    @staticmethod
    def _to_correct_column_format(string):
        return 'risk_{}'.format(convert_camel_case_column_to_snake_case(string.replace('Risk', '')))

    @property
    def schema_out(self):
        schema = {
            **{column: int for column in self.risk_columns},
            **{'time_since_last_{}'.format(column): int for column in self.risk_columns},
            **{'risk_assessment_not_up_to_date': int}
        }

        return schema
