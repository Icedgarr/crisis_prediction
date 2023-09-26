import pandas as pd

from crisis_prediction.features.base import Preprocessor
from crisis_prediction.features.utils import cap_dates


class WellbeingAssessmentPreprocessor(Preprocessor):

    def transform(self, data):
        wellbeing_data = data['wellbeing_screening_table'].copy()
        wellbeing_data['review_period_end_date'] = cap_dates(wellbeing_data['review_period_end_date'])
        wellbeing_data['review_period_start_date'] = pd.to_datetime(wellbeing_data['review_period_start_date'])
        wellbeing_data['review_period_end_date'] = pd.to_datetime(wellbeing_data['review_period_end_date'])
        data['wellbeing_screening_table'] = wellbeing_data.sort_values(['anonymous_pat_id', 'review_period_start_date'])
        return data
