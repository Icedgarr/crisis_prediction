import numpy as np
import pandas as pd
from isoweek import Week

from crisis_prediction.features.base import Feature
from crisis_prediction.features.utils import is_multiindex


class EventFeature(Feature):

    def transform(self, *args, **kwargs):
        pass

    def event_stats_per_week(self, data, patients_first_known_date, column_prefix='event', stats=['sum', 'max', 'min']):
        """
        Input:
            data (pandas.DataFrame): dataframe indexed by
            anonymous_pat_id and a date column and -optionally-
            with extra columns.
            column_prefix (str) [optional]: prefix that will
            be used for the name of the event columns.
            stats (list or dict): set of operations to be applied
            to the dataframe columns.
        Returns:
            event_features: pandas.DataFrame indexed by anonymous_pat_id,
            year and week, the event column and the extra (optional)
            columns aggregated by year and week and summarised by the
            passed statistics.
        """
        events = pd.DataFrame({**{
            'year': [k[1].isocalendar()[0] for k, d in data.iterrows()],
            'week': [k[1].isocalendar()[1] for k, d in data.iterrows()],
            column_prefix: [1] * len(data),
            'anonymous_pat_id': data.index.get_level_values(0)}, **{
            k: data[k] for k in data.columns
        }}).set_index(['anonymous_pat_id', 'year', 'week'])
        patients_first_known_date = patients_first_known_date.isocalendar()
        monday_first_week = Week(patients_first_known_date[0], patients_first_known_date[1]).monday()
        full_dates = pd.date_range(monday_first_week, self.end_date,
                                   freq='W-MON', normalize=True, closed='left')
        full_history = pd.DataFrame({
            'year': [d.isocalendar()[0] for d in full_dates],
            'week': [d.isocalendar()[1] for d in full_dates],
            'anonymous_pat_id': [data.index.get_level_values(0)[0]] * len(full_dates),
        }).set_index(['anonymous_pat_id', 'year', 'week'])
        event_features = full_history.join(events, how='outer').fillna(0) \
            .groupby(['anonymous_pat_id', 'year', 'week']).agg(stats)
        if is_multiindex(event_features.columns):
            event_features.columns = ['_'.join(col).strip() for col in event_features.columns]
        return event_features

    def event_stats_per_t_weeks(self, data, columns_for_t_weeks_stats, t=[4, 8, 12, 16, 20, 24]):
        for col in columns_for_t_weeks_stats.keys():
            t_max = max(t)
            for i in range(1, t_max + 1):
                if '{}_{}_{}'.format(col, i, 'weeks_ago') not in data.columns:
                    data['{}_{}_{}'.format(col, i, 'weeks_ago')] = data[col].shift(i)
            for k in t:
                if '{}_in_last_{}_{}'.format(col, k, 'weeks') not in data.columns:
                    col_list = ['{}_{}_{}'.format(col, i, 'weeks_ago') for i in range(1, k + 1)]
                    data['{}_in_last_{}_{}'.format(col, k, 'weeks')] = data[col_list].agg(
                        columns_for_t_weeks_stats.get(col), axis=1)
        return data

    def event_stats_ever(self, data, columns_for_t_weeks_stats):
        for col in columns_for_t_weeks_stats.keys():
            if '{}_ever'.format(col) not in data.columns:
                data['{}_ever'.format(col)] = data[col].agg('cum' + columns_for_t_weeks_stats.get(col))
        return data

    @staticmethod
    def get_time_since_last_event(events):
        """
        Given a series of True/False values, it returns the number of records since last True.
        It requires the series to be already ordered and be only for each user.
        """
        x1 = (events == 0).cumsum()
        x2 = x1.where(events, np.nan).ffill()
        return (x1 - x2).astype(float)

    @property
    def schema_out(self):
        pass
