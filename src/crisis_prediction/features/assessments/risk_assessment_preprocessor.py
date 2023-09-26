import pandas as pd

from crisis_prediction.features.base import Preprocessor


class RiskAssessmentPreprocessor(Preprocessor):

    def transform(self, data):
        risk_data = data['risk_screening_table']
        risk_data['screening_datetime'] = pd.to_datetime(risk_data['screening_datetime'])
        data['risk_screening_table'] = risk_data.sort_values(['anonymous_pat_id', 'screening_datetime'])
        return data
