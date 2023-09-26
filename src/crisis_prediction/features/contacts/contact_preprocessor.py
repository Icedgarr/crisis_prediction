import pandas as pd

from crisis_prediction.features.base import Preprocessor


class ContactPreprocessor(Preprocessor):

    def transform(self, data):
        data = data.copy()
        data['contacts_table']['contacts_datetime'] = pd.to_datetime(data['contacts_table']['contacts_datetime'])
        data['contacts_table'] = self.create_mapping(data)
        data['contacts_table'] = self.filter_cancelled_appointments(data)
        return data

    def create_mapping(self, data):
        contacts_data = data['contacts_table']
        contacts_data['contact_event_code'] = contacts_data['event_code'] \
            .map(data['contact_eventformat_code']['Category'].to_dict())

        contacts_data['contact_service_code'] = contacts_data['service'] \
            .map(data['service_code']['Category'].to_dict())
        return contacts_data

    def filter_cancelled_appointments(self, data):
        contacts_data = data['contacts_table']
        contacts_data = contacts_data[~((contacts_data['attendance'] == 'Trust cancelled') |
                                        (contacts_data['attendance'] == 'Patient cancelled'))]
        return contacts_data
