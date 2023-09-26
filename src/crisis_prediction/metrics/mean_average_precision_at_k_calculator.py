from numbers import Number
from typing import List, Dict

from numpy import array, mean

from crisis_prediction.metrics.average_precision_at_k_calculator import AveragePrecisionAtKCalculator
from crisis_prediction.metrics.base import AlphaMetric


class MeanAveragePrecisionAtKCalculator(AlphaMetric):
    def __init__(self, k=100):
        self.k = k

    def evaluate(self, targets: array, ranked_list: List[int]) -> Dict[str, Number]:
        sum_average_precision_at_ks = [AveragePrecisionAtKCalculator(k=i + 1).evaluate(targets, ranked_list)[
                                           'average_precision_at_{}'.format(i + 1)] for i in range(self.k)]
        return {'mean_average_precision_at_{}'.format(self.k): mean(sum_average_precision_at_ks)}
