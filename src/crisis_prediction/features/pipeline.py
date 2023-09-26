from typing import List

from crisis_prediction.features.base import Preprocessor, Feature
from crisis_prediction.features.config import TABLE_NAMES


class Pipeline(Feature):
    def __init__(self, preprocessors: List[Preprocessor] = [], features: List[Feature] = []):
        self.features = features
        self.preprocessors = preprocessors
        super(Pipeline, self).__init__()

    def transform(self, data, **kwargs):
        for step in self.preprocessors:
            data = step(data, **kwargs)

        features = {}
        for feature in self.features:
            features[feature.__class__.__name__] = feature(data, **kwargs)

        return features

    @property
    def required_tables(self):
        keys = []
        for transformer in self.preprocessors + self.features:
            keys.extend(list(transformer.schema.keys()))

        return [k for k in keys if k in TABLE_NAMES]

    @property
    def schema_input(self):
        # This is to be improved, currently doesn't really validate the input to the pipeline
        # because the SlotPreprocessor is the first preprocessor
        if not self.preprocessors:
            return {k: v for f in self.features for k, v in f.schema_input.items()}

        return self.preprocessors[0].schema_input

    @property
    def feature_input(self):
        return {k: v for f in self.features for k, v in f.schema_input.items()}

    @property
    def schema_out(self):
        return object
