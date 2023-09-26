from crisis_prediction.features.base import Feature


class StaticFeature(Feature):

    def transform(self, *args, **kwargs):
        pass

    @property
    def schema_out(self):
        pass
