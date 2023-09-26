from datetime import date

import pandas as pd

from crisis_prediction.features.base import Transformer
from crisis_prediction.features.config import START_DATE
from crisis_prediction.features.utils import to_year_month


class ValidUsers(Transformer):
    def __init__(self, minimum_num_crisis=10, minimum_num_days=150, remove_dead=True):
        super(ValidUsers).__init__()
        self.minimum_num_crisis = minimum_num_crisis
        self.minimum_num_days = minimum_num_days
        self.remove_dead = remove_dead

    def transform(self, data, end_date: date = date.today()):
        first_day_month = date(end_date.year, end_date.month, 1)
        patients = data['patient_table']
        crisis_events_table = data['crisis_table'].copy()
        crisis_events_table['event_date'] = pd.to_datetime(crisis_events_table['event_date'])
        crisis_events_table = crisis_events_table[crisis_events_table['event_date'] >= pd.to_datetime(START_DATE)]
        num_crisis = crisis_events_table.groupby('anonymous_pat_id')['event_date'].count()
        valid_patients = list(num_crisis[num_crisis >= self.minimum_num_crisis].index)
        valid_patients_df = patients[patients['anonymous_pat_id'].isin(valid_patients)]
        cond = (first_day_month - valid_patients_df['first_year_month'].apply(
            to_year_month)).dt.days >= self.minimum_num_days
        if self.remove_dead:
            cond = cond & ((valid_patients_df['month_year_death'] >= 290012) | (
                valid_patients_df['month_year_death'].isnull()))
        valid_patients = list(valid_patients_df[cond]['anonymous_pat_id'])
        return valid_patients

    @property
    def schema(self):
        schema = {
            'patient_table': {'anonymous_pat_id': int, 'first_year_month': object, 'month_year_death': object},
            'crisis_table': {'anonymous_pat_id': int, 'event_date': object}
        }
        return schema

    @property
    def schema_out(self):
        return {}
