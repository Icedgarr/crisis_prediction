import numpy as np

from crisis_prediction.features.state_feature import StateFeature
from crisis_prediction.features.utils import convert_camel_case_column_to_snake_case, first_known_to_date


class WellbeingAssessmentStateFeature(StateFeature):

    def transform(self, data):
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['wellbeing_screening_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)

        wellbeing_data = data['wellbeing_screening_table'].copy() \
            .set_index(['anonymous_pat_id', 'review_period_start_date', 'review_period_end_date']).sort_index()
        wellbeing_data.columns = [self._to_correct_column_format(col) for col in wellbeing_data.columns]
        wellbeing_features = self.state_stats_per_week(wellbeing_data, patients_first_known_date)
        return wellbeing_features

    def transform_empty(self, patient_id, patients_first_known_date):
        fields = ['wellbeing_emotional', 'wellbeing_four_factor_total', 'wellbeing_personal',
                  'wellbeing_severe_disturbance', 'wellbeing_social']
        column_value_dict = {column: np.nan for column in fields}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    @staticmethod
    def _to_correct_column_format(string):
        return 'wellbeing_{}'.format(convert_camel_case_column_to_snake_case(string.replace('WellBeing', '')))

    @property
    def schema_out(self):
        fields = ['wellbeing_emotional', 'wellbeing_four_factor_total', 'wellbeing_personal',
                  'wellbeing_severe_disturbance', 'wellbeing_social']
        schema = {column: int for column in fields}
        return schema
