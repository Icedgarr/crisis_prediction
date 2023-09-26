import datetime
from numbers import Number

import numpy as np
import pandas as pd
from isoweek import Week

from crisis_prediction.features import Feature
from crisis_prediction.features.utils import first_known_to_date, dummitize, get_month, difference_in_months


class PatientAgeAndTimeInSystemFeatures(Feature):
    def __init__(self, end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date)
        self.age_bins = [0, 14, 25, 34, 45, 64, 200]
        self.dict_age_bins = {'(0, 14]': 'child', '(15, 25]': 'teen_adult', '(25, 34]': 'young_adult',
                              '(34, 45]': 'middle_age_adult', '(45, 64]': 'older_adult', '(64, 200]': 'elder'}
        self.age_bins_columns = ['current_age_bin_{}'.format(col) for col in self.dict_age_bins.values()]

    def transform(self, data):
        patients_data = data['patient_table']
        if patients_data['month_year_birth'].iloc[0] != patients_data['month_year_birth'].iloc[0]:
            return pd.DataFrame(columns=['anonymous_pat_id', 'year', 'week'] +
                                        list(self.schema_out.keys())).set_index(['anonymous_pat_id', 'year', 'week'])
        patients_data['date_of_birth'] = patients_data['month_year_birth'].apply(
            lambda x: datetime.date(int(str(x)[:4]), int(str(int(x))[4:]), 1) if pd.notnull(x) else np.nan)
        first_known = str(patients_data['first_year_month'].iloc[0])
        patients_first_known_date = first_known_to_date(first_known).isocalendar()
        date_of_birth = patients_data['date_of_birth'].iloc[0]
        patient = patients_data['anonymous_pat_id'].iloc[0]
        monday_first_week = Week(patients_first_known_date[0], patients_first_known_date[1]).monday()
        full_dates = pd.date_range(monday_first_week, self.end_date,
                                   freq='W-MON', normalize=True, closed='left')
        age_features = self.create_age_features(patient, full_dates, date_of_birth)
        features = self.add_time_in_system_features(age_features, first_known)
        return features.drop(columns=['monday_week_year', 'birth_date', 'current_age_bin',
                                      'first_known_year_month_1st', 'year_month_1st'])

    def create_age_features(self, patient, full_dates, date_of_birth):
        age_features = pd.DataFrame(
            {'anonymous_pat_id': patient, 'monday_week_year': full_dates, 'birth_date': pd.to_datetime(date_of_birth)})
        age_features['current_age'] = (age_features['monday_week_year'].dt.year -
                                       age_features['birth_date'].dt.year - (age_features['monday_week_year'].dt.month <
                                                                             age_features['birth_date'].dt.month))
        age_features['current_age_bin'] = pd.cut(x=age_features['current_age'], bins=self.age_bins)
        age_features['current_age_bin'] = age_features['current_age_bin'].astype(str).replace(self.dict_age_bins)
        age_features['older_than_65'] = (age_features['current_age_bin'] == 'elder').astype(int)
        age_features = age_features.join(dummitize(age_features['current_age_bin'], self.age_bins_columns))
        return age_features

    def add_time_in_system_features(self, features, first_known):
        features['year'] = features['monday_week_year'].apply(lambda d: d.isocalendar()[0])
        features['week'] = features['monday_week_year'].apply(lambda d: d.isocalendar()[1])
        features['years_since_known'] = features['year'] - int(str(first_known)[:4])
        features['first_known_year_month_1st'] = datetime.datetime(int(str(first_known)[:4]),
                                                                   int(str(first_known)[-2:]), 1)

        features['year_month_1st'] = features.apply(
            lambda row: datetime.datetime(row['year'], get_month(row['week']), 1), axis=1)

        features['months_since_known'] = features.apply(
            lambda row: max(0, difference_in_months(row['first_known_year_month_1st'],
                                                    row['year_month_1st'])), axis=1).astype(int)

        return features.set_index(['anonymous_pat_id', 'year', 'week'])

    @property
    def schema_out(self):
        schema = {
            **{column: Number for column in self.age_bins_columns},
            **{'current_age': Number,
               'older_than_65': Number,
               'years_since_known': Number,
               'months_since_known': Number}
        }
        return schema
