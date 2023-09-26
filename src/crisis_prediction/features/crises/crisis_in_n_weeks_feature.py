import datetime

import pandas as pd

from crisis_prediction.features.event_feature import EventFeature


class CrisisInNWeeksFeature(EventFeature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), n=4):
        super().__init__(end_date)
        self.n = n

    def transform(self, data):
        """This function takes a dictionary of dataframes
        with the table names as keys and returns features
        for crisis.
        """
        crisis_data = data['InCrisisPeriod'].sort_index()['crisis_burst_1week']
        crisis_features = self.crisis_in_n_weeks(crisis_data).rename('crisis_in_{}_weeks'.format(self.n))
        return pd.DataFrame(crisis_features).fillna(0)

    def crisis_in_n_weeks(self, crisis_data):
        return pd.concat([crisis_data.shift(-i) for i in range(1, self.n + 1)], axis=1).max(axis=1, skipna=False)

    @property
    def schema_out(self):
        schema = {
            'crisis_in_{}_weeks'.format(self.n): float,
        }
        return schema
