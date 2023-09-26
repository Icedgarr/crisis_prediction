import pandas as pd
from isoweek import Week

from crisis_prediction.features.base import Feature


class StateFeature(Feature):

    def transform(self, data):
        pass

    def state_stats_per_week(self, data, patients_first_known_date, argument='last'):
        """This function takes the a dataframe indexed by
        (patient_id, starting_date, ending_date) and some
        columns returns a dataframe with the full story
        of a patient index by anonymous_pat_id, date, and the
        corresponding features for the state.
        """
        extended_state = pd.concat([self.get_extended_state_lambda(k, v) for k, v in data.iterrows()]) \
            .set_index(['anonymous_pat_id', 'year', 'week'])
        patients_first_known_date = patients_first_known_date.isocalendar()
        monday_first_week = Week(patients_first_known_date[0], patients_first_known_date[1]).monday()
        full_history = self.get_full_history(data, monday_first_week)
        extended_state = extended_state.join(full_history, how='outer')
        state_features = extended_state.groupby(['anonymous_pat_id', 'year', 'week']).agg(argument)
        return state_features

    def get_full_history(self, data, monday_first_week):
        full_dates = pd.date_range(monday_first_week, self.end_date, freq='W-MON',
                                   normalize=True, closed='left')
        full_history = pd.DataFrame({
            'anonymous_pat_id': [data.index.get_level_values(0)[0]] * len(full_dates),
            'year': [d.isocalendar()[0] for d in full_dates],
            'week': [d.isocalendar()[1] for d in full_dates],
        }).set_index(['anonymous_pat_id', 'year', 'week'])
        return full_history

    def get_extended_state_lambda(self, index, row):
        """This function takes a multi_index (patient_id,
        starting_date, end_date) of a row as well as its
        row and returns a dataframe filled with the date
        in between those dates and the corresponding features."""
        starting_date = index[1].isocalendar()
        monday_starting_date = Week(starting_date[0], starting_date[1]).monday()
        end_date = index[2].isocalendar()
        monday_end_date = Week(end_date[0], end_date[1]).monday()
        dates = pd.date_range(monday_starting_date, min(monday_end_date, self.end_date), freq='W-MON', normalize=True)
        if dates.empty:
            dates = [index[1]]

        extended_state = pd.DataFrame({
            **{
                'year': [d.isocalendar()[0] for d in dates],
                'week': [d.isocalendar()[1] for d in dates],
                'anonymous_pat_id': [index[0]] * len(dates),
            },
            **{k: row[k] for k in row.index},
        })
        return extended_state

    @property
    def schema_out(self):
        pass
