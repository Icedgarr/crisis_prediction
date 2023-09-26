import datetime

from crisis_prediction.features.state_feature import StateFeature
from crisis_prediction.features.utils import dummitize, first_known_to_date, convert_camel_case_column_to_snake_case


class MhaEpisodeStateFeatures(StateFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date=end_date)
        self.source_columns = ['cto_status', 'on_conditional_discharge', 'mha_section_code']
        self.cto_categories = ['Active', 'Not applicable', 'Recalled']
        self.cto_status_columns = ['cto_status_{}'.format(c)
                                   for c in self.cto_categories]

    def transform(self, data):
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['mha_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        mha_data = data['mha_table'].set_index(['anonymous_pat_id', 'start_date_time', 'end_date_time'])
        mha_data.columns = [convert_camel_case_column_to_snake_case(col) for col in self.source_columns]
        mha_data['mha_section_code'] = mha_data['mha_section_code'].apply(lambda x: 0 if x == 'Inf' else 1)
        mha_data[self.cto_status_columns] = dummitize(mha_data['cto_status'], self.cto_status_columns)
        mha_data.rename(columns={col: convert_camel_case_column_to_snake_case(col) for col in self.cto_status_columns},
                        inplace=True)
        mha_features = self.state_stats_per_week(mha_data[mha_data.columns[1:]],
                                                 patients_first_known_date).fillna(0)
        return mha_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {column: 0 for column in self._get_out_column_names()}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def _get_out_column_names(self):
        fields = [convert_camel_case_column_to_snake_case(col) for col in
                  self.source_columns[1:] + self.cto_status_columns]
        return fields

    @staticmethod
    def _to_correct_column_format(string):
        return '{}'.format(convert_camel_case_column_to_snake_case(string))

    @property
    def schema_out(self):
        schema = {column: int for column in self._get_out_column_names()}
        return schema
