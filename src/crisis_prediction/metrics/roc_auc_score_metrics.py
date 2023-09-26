from numbers import Number
from typing import Dict

from sklearn.metrics import roc_auc_score

from crisis_prediction.metrics.base import AlphaMetric


class RocAucScoreProbabilityMetric(AlphaMetric):
    def evaluate(self, targets, prediction_probabilities) -> Dict[str, Number]:
        return {'roc_auc': roc_auc_score(targets, prediction_probabilities)}


class RocAucScoreRankingMetric(AlphaMetric):
    @staticmethod
    def _reverse_ranking(predicted_ranking):
        # Make the top ranking be the highest numbers as that's the way roc_auc expects it
        return len(predicted_ranking) - predicted_ranking

    def evaluate(self, targets, predicted_ranking) -> Dict[str, Number]:
        reverse_ranking = self._reverse_ranking(predicted_ranking)
        return {'roc_auc': roc_auc_score(targets, reverse_ranking)}
