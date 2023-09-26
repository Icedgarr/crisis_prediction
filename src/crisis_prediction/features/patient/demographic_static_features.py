import pandas as pd

from crisis_prediction.features.static_feature import StaticFeature
from crisis_prediction.features.utils import convert_camel_case_column_to_snake_case, add_missing_columns


class DemographicStaticFeatures(StaticFeature):
    dummy_columns = ['static_gender_female', 'static_gender_male', 'static_ethnic_group_asian',
                     'static_ethnic_group_black', 'static_ethnic_group_mixed', 'static_ethnic_group_not_known',
                     'static_ethnic_group_other', 'static_ethnic_group_white',
                     'static_marital_status_co_habitee', 'static_marital_status_divorced',
                     'static_marital_status_married', 'static_marital_status_not_disclosed',
                     'static_marital_status_not_asked', 'static_marital_status_not_recorded',
                     'static_marital_status_other_unknown', 'static_marital_status_separated',
                     'static_marital_status_single', 'static_marital_status_unknown',
                     'static_marital_status_widowed']
    date_columns = ['static_first_contact_year', 'static_first_contact_month',
                    'static_death_year', 'static_death_month',
                    'static_birth_year']

    def transform(self, data):
        """
        Take Patients dataframe and extract the static features and characteristics of the patients.
        """
        patients_data = data['patient_table'].set_index('anonymous_pat_id')
        static_features = pd.get_dummies(patients_data[['Gender', 'ethnicity', 'marital_status']])
        static_features.columns = ['static_{}'.format(convert_camel_case_column_to_snake_case(col))
                                   for col in static_features.columns]
        static_features = add_missing_columns(static_features, self.dummy_columns).fillna(0).astype(int)
        static_features['static_death_year'] = patients_data['month_year_death'] \
            .apply(lambda x: None if x is None else int(str(int(x))[:4]))
        static_features['static_death_month'] = patients_data['month_year_death'] \
            .apply(lambda x: None if x is None else int(str(int(x))[-2:]))
        static_features['static_birth_year'] = patients_data['month_year_birth'] \
            .apply(lambda x: int(str(x)[:4]))
        static_features['static_first_contact_year'] = patients_data['first_year_month'] \
            .apply(lambda x: int(str(x)[:4]))
        static_features['static_first_contact_month'] = patients_data['first_year_month'] \
            .apply(lambda x: int(str(x)[-2:]))
        return static_features

    @property
    def schema_out(self):
        schema = {column: int for column in self.dummy_columns + self.date_columns}
        return schema
