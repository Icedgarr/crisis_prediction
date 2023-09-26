from crisis_prediction.features.base import Preprocessor
import re


class TextEmbeddingPreprocessor(Preprocessor):

    def transform(self, data):
        data['clinical_notes_table'] = self.add_processed_text_column(data['clinical_notes_table'])
        return data

    def add_processed_text_column(self, data):
        data['processed_anonymized_text'] = data['anonymized_text'].apply(self.preprocess_text)
        return data

    @staticmethod
    def preprocess_text(txt):
        txt = re.sub(r'(&nbsp;)', ' ', txt)
        txt = re.sub(r'<[^>]*>', ' ', txt)
        txt = re.sub(r'\s+', ' ', txt)
        return txt
