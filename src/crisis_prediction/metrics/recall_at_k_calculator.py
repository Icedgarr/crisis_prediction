from numbers import Number
from typing import List, Dict

import pandas as pd
from numpy import array

from crisis_prediction.metrics.base import AlphaMetric


class RecallAtKCalculator(AlphaMetric):
    def __init__(self, k=100):
        self.k = k

    def evaluate(self, targets: array, ranked_list: List[int]) -> Dict[str, Number]:
        first_k_targets = self.get_first_k_targets(targets, ranked_list)
        return {'recall_at_{}'.format(self.k): first_k_targets.sum() / targets.sum()}

    def get_first_k_targets(self, targets: array, ranked_list: List[int]):
        targets = pd.Series(targets)
        targets.index = ranked_list
        sorted_targets = targets.sort_index()
        first_k_targets = sorted_targets.values[:self.k]
        return first_k_targets
