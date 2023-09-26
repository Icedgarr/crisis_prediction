import pandas as pd

from crisis_prediction.features.base import Preprocessor


class CrisisPreprocessor(Preprocessor):

    def transform(self, data):
        data['crisis_table'] = self.add_date_columns(data['crisis_table'])
        data['crisis_table'] = data['crisis_table'].sort_values(['anonymous_pat_id', 'event_date'])
        return data

    @staticmethod
    def add_date_columns(data):
        data['event_date'] = pd.to_datetime(data['event_date'])
        data['year'] = data['event_date'].dt.year
        data['week'] = data['event_date'].dt.week
        return data
