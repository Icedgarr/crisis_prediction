import datetime

from crisis_prediction.features.base import Feature


class InCrisisPeriod(Feature):
    def __init__(self, end_date: datetime.date = datetime.date.today(), weeks_before_new_burst=1):
        """
        :param weeks_before_new_burst: number of weeks to pass without a crisis to consider that the patient is not in
        the crisis period anymore. It has to be an integer
        """
        super().__init__(end_date)
        self.weeks_before_new_burst = weeks_before_new_burst

    def transform(self, data):
        """
        :param data: Dictionary of dataframes that contains the key 'CrisisEvents' that has as value a dataframe
        with at least the next columns:
        ['anonymous_pat_id' (int), 'event_date' (object)]
        :return:
        """
        crisis_event_features = data['CrisisEventFeatures'].copy()
        crisis_event_features = self.add_crisis_burst(crisis_event_features)
        crisis_event_features = self.add_in_crisis_period(crisis_event_features)

        columns_to_return = ['crisis_burst_{}week'.format(self.weeks_before_new_burst),
                             'number_crisis_burst_{}week'.format(self.weeks_before_new_burst),
                             'in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst),
                             'number_crises_burst_{}week_passed'.format(self.weeks_before_new_burst)]
        return crisis_event_features[columns_to_return].astype(int)

    def add_crisis_burst(self, data):
        data['crisis_burst_{}week'.format(self.weeks_before_new_burst)] = 0
        data.loc[(data['crisis_max'] == 1) &
                 ((data['time_since_last_crisis'].shift(1) >= self.weeks_before_new_burst) |
                  (data['time_since_last_crisis'].shift(1).isnull())),
                 'crisis_burst_{}week'.format(self.weeks_before_new_burst)] = 1
        return data

    def add_in_crisis_period(self, data):
        bursts = data.loc[data['crisis_burst_{}week'.format(self.weeks_before_new_burst)] == 1]

        data.loc[data['crisis_burst_{}week'.format(self.weeks_before_new_burst)] == 1,
                 'number_crisis_burst_{}week'.format(self.weeks_before_new_burst)] = range(1, len(bursts) + 1)

        data.loc[(data['time_since_last_crisis'] == self.weeks_before_new_burst) |
                 (data['time_since_last_crisis'].isnull()),
                 'number_crisis_burst_{}week'.format(self.weeks_before_new_burst)] = 0.

        data['number_crisis_burst_{}week'.format(self.weeks_before_new_burst)].ffill(inplace=True)

        data['in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst)] = 1.

        data.loc[data['number_crisis_burst_{}week'.format(self.weeks_before_new_burst)] == 0,
                 'in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst)] = 0.
        data['number_crises_burst_{}week_passed'.format(self.weeks_before_new_burst)] = data[
            'crisis_burst_{}week'.format(self.weeks_before_new_burst)].cumsum()
        return data

    @property
    def schema_out(self):
        columns_to_return = ['crisis_burst_{}week'.format(self.weeks_before_new_burst),
                             'number_crisis_burst_{}week'.format(self.weeks_before_new_burst),
                             'in_crisis_period_burst_{}week'.format(self.weeks_before_new_burst),
                             'number_crises_burst_{}week_passed'.format(self.weeks_before_new_burst)]
        schema = {k: int for k in columns_to_return}
        return schema
