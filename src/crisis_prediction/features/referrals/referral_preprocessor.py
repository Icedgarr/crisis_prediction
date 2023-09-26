from crisis_prediction.features.base import Preprocessor


class ReferralPreprocessor(Preprocessor):

    def transform(self, data):
        data['referral_table'] = self.create_mappings(data)
        return data

    @staticmethod
    def create_mappings(data):
        service_code = {k: (1 if v == 'Planned' else 0) for k, v in data['service_code'].to_dict()['Category'].items()}
        discharge_code = data['discharge_code'].to_dict()['Category']
        source_code = data['source_code'].to_dict()['Category']
        referral_data = data['referral_table']
        referral_data['service_planned'] = referral_data['service'].map(service_code)
        referral_data['discharge_category'] = referral_data['discharge_reason'].map(discharge_code)
        referral_data['source_category'] = referral_data['referral_source'].map(source_code)
        return referral_data
