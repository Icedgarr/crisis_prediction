import datetime
from abc import ABCMeta, abstractmethod

import pandas as pd
from isoweek import Week

from crisis_prediction.features.exceptions import SchemaException


class Transformer(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @staticmethod
    def apply_schemata(func):
        def new_func(self, data):
            schema = self.schema
            data_needed = data.copy()
            for table in schema.keys():
                if table not in data.keys():
                    raise SchemaException('{} DataFrame not passed to {}'.format(table, self.__class__.__name__))
                elif data[table].empty:
                    data_needed[table] = data[table]
                    # raise SchemaException('{} DataFrame passed to {} was empty'.format(table, self.__class__.__name__))
                elif any(c not in data[table].columns for c in schema[table]):
                    missing_column = next(c for c in schema[table] if c not in data[table].columns)
                    raise SchemaException('{} DataFrame passed to {} missing the {} column'
                                          .format(table, self.__class__.__name__, missing_column))
                else:
                    data_needed[table] = data[table][[*schema[table].keys()]]

            return func(self, data_needed)

        return new_func

    @staticmethod
    def check_schemata(func):
        def new_func(self, data):
            schema = self.schema
            for table in schema.keys():
                if table not in data.keys():
                    raise SchemaException('{} DataFrame not passed to {}'.format(table, self.__class__.__name__))
                elif data[table].empty:
                    data[table] = pd.DataFrame(columns=list(schema[table].keys()))
                elif any(c not in data[table].columns for c in schema[table]):
                    missing_column = next(c for c in schema[table] if c not in data[table].columns)
                    raise SchemaException('{} DataFrame passed to {} missing the {} column'
                                          .format(table, self.__class__.__name__, missing_column))

            return func(self, data)

        return new_func

    @staticmethod
    def apply_schemata_out(func):
        def new_func(self, data):
            output = func(self, data)
            schema = self.schema_out
            if any(c not in output.columns for c in schema):
                missing_column = next(c for c in schema if c not in output.columns)
                raise SchemaException('Output of {} missing the {} column'
                                      .format(self.__class__.__name__, missing_column))
            else:
                output = output[list(schema.keys())]
            return output

        return new_func

    @staticmethod
    def check_input_flat_schema(func):
        def new_func(self, data):
            if len(data.keys()) < 1:
                raise SchemaException('Empty data dict passed to {}'.format(self.__class__.__name__))

            for table in data.keys():
                if any(k not in data[table].keys() for k in self.schema.keys()):
                    missing_column = next(c for c in self.schema.keys() if c not in data[table].columns)
                    raise SchemaException('{} DataFrame passed to {} missing the {} column'
                                          .format(table, self.__class__.__name__, missing_column))
                elif data[table].empty:
                    raise SchemaException('{} DataFrame passed to {} was empty'.format(table, self.__class__.__name__))

            return func(self, data)

        return new_func

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass


class Preprocessor(Transformer):
    def __init__(self):
        super(Preprocessor, self).__init__()

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass


class Feature(Transformer):
    def __init__(self, end_date: datetime.date = None):
        super(Feature, self).__init__()
        self.end_date = self.end_date = datetime.date.today() if end_date is None else end_date

    def create_features_empty_data(self, patient_id, patients_first_known_date, column_value_dict):
        patients_first_known_date = patients_first_known_date.isocalendar()
        monday_first_week = Week(patients_first_known_date[0], patients_first_known_date[1]).monday()
        full_dates = pd.date_range(monday_first_week, self.end_date,
                                   freq='W-MON', normalize=True, closed='left')
        full_history = pd.DataFrame({
            'year': [d.isocalendar()[0] for d in full_dates],
            'week': [d.isocalendar()[1] for d in full_dates],
            'anonymous_pat_id': [patient_id] * len(full_dates)
        }).set_index(['anonymous_pat_id', 'year', 'week'])
        for key, value in column_value_dict.items():
            full_history[key] = value
        return full_history

    @property
    @abstractmethod
    def schema_out(self):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
