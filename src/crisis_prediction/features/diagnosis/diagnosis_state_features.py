import datetime

import numpy as np

from crisis_prediction.features.diagnosis.aux_diagnosis_columns import granular_category_columns, broad_category_columns
from crisis_prediction.features.state_feature import StateFeature
from crisis_prediction.features.utils import first_known_to_date


class DiagnosisStateFeatures(StateFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date=end_date)
        self.granular_categories = granular_category_columns
        self.broad_category = broad_category_columns
        self.state_columns = self.granular_categories + self.broad_category
        self.fields = ['current_' + col for col in self.state_columns]
        self.fields += ['ever_' + col.replace('diagnosis', 'diagnosed') for col in self.state_columns]
        self.fields += ['number_of_current_broad_diagnosis', 'number_of_current_granular_diagnosis',
                        'current_dual_diagnosis', 'ever_dual_diagnosis', 'number_of_ever_broad_diagnosis',
                        'number_of_ever_granular_diagnosis']

    def transform(self, data):
        """This function takes a dictionary of dataframes
        with the table names as keys and returns features
        for diagnosis.
        """
        first_known = str(data['patient_table']['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known)
        if data['diagnosis_table'].empty:
            patient_id = data['patient_table']['anonymous_pat_id'].iloc[0]
            return self.transform_empty(patient_id, patients_first_known_date)
        diagnosis_data = data['diagnosis_table'].set_index(
            ['anonymous_pat_id', 'diagnosis_start_date', 'diagnosis_end_date'])

        diagnosis_features = self.create_current_diagnosis_features(diagnosis_data, patients_first_known_date)
        diagnosis_features = self.add_number_of_current_diagnosis_features(diagnosis_features)
        diagnosis_features = self.add_dual_diagnosis_feature(diagnosis_features)
        diagnosis_features = self.add_ever_diagnosed_features(diagnosis_features)

        return diagnosis_features

    def transform_empty(self, patient_id, patients_first_known_date):
        column_value_dict = {column: 0 for column in self.fields}
        return self.create_features_empty_data(patient_id, patients_first_known_date, column_value_dict)

    def create_current_diagnosis_features(self, data, patients_first_known_date):
        diagnosis_features = self.state_stats_per_week(data[self.granular_categories + self.broad_category],
                                                       patients_first_known_date).fillna(0)
        diagnosis_features.rename(columns={col: 'current_' + col for col in self.state_columns},
                                  inplace=True)
        return diagnosis_features

    def add_number_of_current_diagnosis_features(self, data):
        broad_diagnosis_columns = ['current_' + col for col in self.state_columns if 'broad' in col]
        data['number_of_current_broad_diagnosis'] = data[broad_diagnosis_columns].sum(axis=1)
        granular_diagnosis_columns = ['current_' + col for col in self.state_columns if 'granular' in col]
        data['number_of_current_granular_diagnosis'] = data[granular_diagnosis_columns].sum(axis=1)
        return data

    def add_dual_diagnosis_feature(self, data):
        psychological_diagnosis_columns = ['current_' + col for col in self.broad_category if
                                           ('non_psycho' not in col)]
        data['current_dual_diagnosis'] = ((data[psychological_diagnosis_columns].sum(axis=1) >= 2) &
                                          (data['current_diagnosis_broad_substance_misuse'] == 1)).astype(int)
        return data

    def add_ever_diagnosed_features(self, data):
        current_diagnosis_columns = ['current_dual_diagnosis'] + ['current_' + col for col in self.state_columns]
        ever_diagnosis_columns = ['ever_dual_diagnosis'] + ['ever_' + col.replace('diagnosis', 'diagnosed') for col in
                                                            self.state_columns]
        data[ever_diagnosis_columns] = data[current_diagnosis_columns].replace({0: np.nan}).ffill().fillna(0)

        broad_diagnosis_columns = [col for col in ever_diagnosis_columns if 'broad' in col]
        data['number_of_ever_broad_diagnosis'] = data[broad_diagnosis_columns].sum(axis=1)

        granular_diagnosis_columns = [col for col in ever_diagnosis_columns if 'granular' in col]
        data['number_of_ever_granular_diagnosis'] = data[granular_diagnosis_columns].sum(axis=1)
        return data

    @property
    def schema_out(self):
        schema = {**{column: int for column in self.fields}}
        return schema
