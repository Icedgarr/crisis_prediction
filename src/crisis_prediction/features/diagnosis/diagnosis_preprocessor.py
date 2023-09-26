import pandas as pd

from crisis_prediction.features.base import Preprocessor
from crisis_prediction.features.diagnosis.aux_diagnosis_columns import granular_category_columns, broad_category_columns


class DiagnosisPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()
        self.diagnosis_code_columns = ['Diagnosis01Code', 'Diagnosis02Code', 'Diagnosis03Code', 'Diagnosis04Code',
                                       'Diagnosis05Code', 'Diagnosis06Code', 'Diagnosis07Code', 'Diagnosis08Code',
                                       'Diagnosis09Code', 'Diagnosis10Code', 'Diagnosis11Code', 'Diagnosis12Code',
                                       'Diagnosis13Code', 'Diagnosis14Code']

    def transform(self, data):
        diagnosis_data = data['diagnosis_table'].copy()
        diagnosis_data = self.create_mappings(diagnosis_data, data['diagnosis_broad_codes'], 'diagnosis_broad')
        diagnosis_data = self.create_mappings(diagnosis_data, data['diagnosis_granular_codes'], 'diagnosis_granular')
        data['diagnosis_table'] = diagnosis_data
        return data

    def create_mappings(self, diagnosis_data, diagnosis_codes, root):
        for category in diagnosis_codes['Category'].unique():
            root_codes = set(diagnosis_codes.loc[diagnosis_codes['Category'] == category, 'DiagnosisCodeRoot'])
            diagnosis_data['{}_{}'.format(root, category.lower())] = pd.concat(
                [diagnosis_data[col].str.contains(code) for col in self.diagnosis_code_columns for code in root_codes],
                axis=1).any(axis=1).astype(int)
        return diagnosis_data

    @property
    def schema_out(self):
        schema = {
            'diagnosis_table': {**{
                'anonymous_pat_id': int,
                'diagnosis_start_date': object,
                'diagnosis_end_date': object
            }, **{
                col: int for col in broad_category_columns
            }, **{
                col: int for col in granular_category_columns
            }}
        }
        return schema
