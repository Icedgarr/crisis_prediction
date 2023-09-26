import datetime

from crisis_prediction.features.state_feature import StateFeature
from crisis_prediction.features.utils import dummitize, first_known_to_date, convert_camel_case_column_to_snake_case


class ReferralStateFeatures(StateFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date=end_date)
        self.source_categories = ['Acute', 'Ambulance', 'Carer', 'Community', 'GP',
                                  'Internal', 'Local Authority', 'Mental_Health',
                                  'Other Agency', 'Primary Care', 'Self']
        self.source_columns = ['referral_state_source_category_{}'.format(c) for c in self.source_categories]

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
        referral_data = data['referral_table'].set_index(['anonymous_pat_id', 'referral_date', 'discharge_date'])
        referral_data[self.source_columns] = dummitize(referral_data['source_category']
                                                       .rename('referral_state_source_category'),
                                                       self.source_columns)
        referral_features = self.state_stats_per_week(referral_data[self.source_columns],
                                                      patients_first_known_date, argument='max').fillna(0)
        referral_features.columns = [convert_camel_case_column_to_snake_case(c)
                                     for c in referral_features.columns]
        return referral_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {convert_camel_case_column_to_snake_case(c): 0 for c in self.source_columns}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    @property
    def schema_out(self):
        fields = [convert_camel_case_column_to_snake_case(c) for c in self.source_columns]
        schema = {column: int for column in fields}
        return schema
