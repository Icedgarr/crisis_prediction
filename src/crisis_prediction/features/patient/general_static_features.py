import datetime
from numbers import Number
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from crisis_prediction.features.static_feature import StaticFeature
from crisis_prediction.features.utils import difference_in_months


class GeneralStaticFeatures(StaticFeature):

    def transform(self, data: Dict[str, DataFrame]):
        patients_data = data['patient_table'].set_index('anonymous_pat_id').copy()
        contacts_data = data['contacts_table'].copy()
        crisis_data = data['crisis_table'].copy()
        features = self.create_patient_features(patients_data)
        features = self.add_last_contact_feature(features, contacts_data)
        features = self.add_crisis_features(features, crisis_data)

        return features

    def create_patient_features(self, patients_data):
        patients_data['month_year_death'] = patients_data['month_year_death'].fillna('299901')
        features = patients_data[['first_year_month', 'month_year_death']]
        features['first_known_year_month_1st'] = features['first_year_month'].apply(
            lambda first_known: datetime.datetime(int(str(first_known)[:4]),
                                                  int(str(first_known)[-2:]), 1))
        features['last_updated'] = pd.to_datetime(self.end_date)
        features['days_in_system_to_date'] = ((features['last_updated'] -
                                               features['first_known_year_month_1st']) /
                                              np.timedelta64(1, 'D')).astype(int)
        features['months_in_system_to_date'] = features.apply(
            lambda row: max(0, difference_in_months(row['first_known_year_month_1st'],
                                                    row['last_updated'])), axis=1).astype(int)
        features['death_year_month_1st'] = features['month_year_death'].apply(
            lambda death_date: datetime.datetime(min(2200, int(str(death_date)[:4])),
                                                 int(str(death_date)[-2:]), 1))
        features['alive_to_date'] = (features['death_year_month_1st'] > features['last_updated']).astype(int)
        return features.drop(columns=['death_year_month_1st', 'first_known_year_month_1st'])

    def add_last_contact_feature(self, features, contacts_data):
        if contacts_data.empty:
            features['last_contact_to_date'] = np.nan
            features['days_since_last_contact_to_date'] = np.nan
            features['months_since_last_contact_to_date'] = np.nan
            return features
        contacts_data['contacts_datetime'] = pd.to_datetime(contacts_data['contacts_datetime'])
        last_contact = contacts_data.groupby('anonymous_pat_id')['contacts_datetime'].max()
        last_contact.name = 'last_contact_to_date'
        features = features.join(last_contact)
        features['days_since_last_contact_to_date'] = ((features['last_updated'] -
                                                        features['last_contact_to_date']) /
                                                       np.timedelta64(1, 'D')).astype(int)
        features['months_since_last_contact_to_date'] = features.apply(
            lambda row: max(0, difference_in_months(row['last_contact_to_date'],
                                                    row['last_updated'])), axis=1).astype(int)
        return features

    def add_crisis_features(self, features, crisis_data):
        if crisis_data.empty:
            features['last_crisis_to_date'] = np.nan
            features['number_of_crisis_events_to_date'] = 0
            features['days_since_last_crisis_to_date'] = np.nan
            features['months_since_last_crisis_to_date'] = np.nan
            return features
        crisis_data['event_date'] = pd.to_datetime(crisis_data['event_date'])
        crisis_features = crisis_data.groupby('anonymous_pat_id').agg({'event_date': 'max', 'CrisisEventID': 'count'})
        crisis_features.columns = ['last_crisis_to_date', 'number_of_crisis_events_to_date']
        features = features.join(crisis_features)
        features['days_since_last_crisis_to_date'] = ((features['last_updated'] -
                                                       features['last_crisis_to_date']) /
                                                      np.timedelta64(1, 'D')).astype(int)
        features['months_since_last_crisis_to_date'] = features.apply(
            lambda row: max(0, difference_in_months(row['last_crisis_to_date'],
                                                    row['last_updated'])), axis=1).astype(int)
        return features

    @property
    def schema_out(self):
        schema = {
            'last_updated': object,
            'days_in_system_to_date': Number,
            'months_in_system_to_date': Number,
            'alive_to_date': Number,
            'last_contact_to_date': object,
            'days_since_last_contact_to_date': Number,
            'months_since_last_contact_to_date': Number,
            'last_crisis_to_date': object,
            'number_of_crisis_events_to_date': Number,
            'days_since_last_crisis_to_date': Number,
            'months_since_last_crisis_to_date': Number
        }
        return schema
