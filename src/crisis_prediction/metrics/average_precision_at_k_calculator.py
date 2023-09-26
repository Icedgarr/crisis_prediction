from numbers import Number
from typing import List, Dict

from numpy import array, sum

from crisis_prediction.metrics.base import AlphaMetric
from crisis_prediction.metrics.precision_at_k_calculator import PrecisionAtKCalculator


class AveragePrecisionAtKCalculator(AlphaMetric):
    def __init__(self, k=100):
        self.k = k

    def evaluate(self, targets: array, ranked_list: List[int]) -> Dict[str, Number]:
        relevance_at_ks = PrecisionAtKCalculator(k=self.k).get_first_k_targets(targets, ranked_list)
        relevant_precision_at_ks = [PrecisionAtKCalculator(k=i + 1).evaluate(targets, ranked_list)[
                                        'precision_at_{}'.format(i + 1)]
                                    if relevance_at_ks[i] != 0 else 0 for i in range(len(relevance_at_ks))]
        return {'average_precision_at_{}'.format(self.k): sum(relevant_precision_at_ks) / sum(relevance_at_ks) if sum(
            relevance_at_ks) != 0 else 0}
