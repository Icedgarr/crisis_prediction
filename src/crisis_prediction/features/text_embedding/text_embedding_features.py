import os, datetime
import torch
import numpy as np
from crisis_prediction.features.base import Feature
from crisis_prediction.features.utils import isocalendar_week, isocalendar_year
from transformers import DistilBertModel, DistilBertTokenizer


class TextEmbeddingFeatures(Feature):
    def __init__(self, device_name: str = 'cuda', cuda_devices: str = '1',
                 end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date=end_date)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        self.device = torch.device(device_name)
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert_model.to(self.device)

    def transform(self, data):
        """
        Takes a dictionary of dataframes and returns the text features related to ProgressNote indexed by patient,
        year and week.
        """
        progress_notes_data = data['clinical_notes_table']
        progress_notes_data = self.add_text_embedding_column(progress_notes_data)
        progress_notes_data['year'] = progress_notes_data['entered_datetime'].apply(isocalendar_year)
        progress_notes_data['week'] = progress_notes_data['entered_datetime'].apply(isocalendar_week)
        return progress_notes_data

    def add_text_embedding_column(self, data):
        data['Embeddinganonymized_text'] = data['processed_anonymized_text'].apply(lambda doc:
                                                                                   self.bert_sentence_embedding(
                                                                                       self.bert_model,
                                                                                       self.bert_tokenizer,
                                                                                       doc))
        return data

    def bert_sentence_embedding(self, model, tokenizer, txt):
        tokens = tokenizer.encode(txt, max_length=512)
        inputs = torch.tensor([tokens])
        states = model(inputs.to(self.device))
        return states[0].detach().cpu().numpy().mean(axis=1)[0]

    @property
    def schema_out(self):
        schema = {
            'Embeddinganonymized_text': np.array,
            'year': int,
            'week': int,
            'anonymous_pat_id': int,
            'entered_datetime': str,
        }
        return schema
