from typing import List, Dict
from numbers import Number

from pandas import Series
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score, recall_score, precision_score, \
    f1_score

from crisis_prediction.metrics.average_precision_at_k_calculator import AveragePrecisionAtKCalculator, \
    PrecisionAtKCalculator
from crisis_prediction.metrics.precision_at_k_calculator import PrecisionAtKCalculator
from crisis_prediction.metrics.base import AlphaMetric


class ComputeGroupMetrics:
    def __init__(self,
                 metric_calculators: List[AlphaMetric],
                 group_column: str = 'week_number',
                 target_column: str = 'crisis_in_4_weeks',
                 predictions_column: str = 'predictions',
                 threshold: float = 0.5):
        self.group_column = group_column
        self.target_column = target_column
        self.predictions_column = predictions_column
        self.metric_calculators = metric_calculators
        self.threshold = threshold

    def get_metrics(self, predictions_targets) -> Dict[str, List[Number]]:
        metrics = predictions_targets.groupby(self.group_column).apply(
            lambda subset: self.evaluate_metrics(subset[self.target_column], subset[self.predictions_column]))
        metrics = metrics.reset_index().to_dict(orient='list')
        return metrics

    def evaluate_metrics(self, targets: Series, predictions: Series) -> Series:
        metrics_dict = self.compute_binary_metrics(targets, predictions, flag_average_precision=True,
                                                   threshold=self.threshold)
        return pd.Series(metrics_dict)

    def compute_binary_metrics(self, target, predictions_proba, flag_average_precision, threshold=0.5):
        metrics = self.compute_metrics_of_confusion_matrix(target, predictions_proba > threshold)
        metrics['roc_auc'] = self.compute_roc_auc_score(target, predictions_proba)
        metrics['kappa'] = cohen_kappa_score(target, predictions_proba.round())
        metrics['precision_at_100'] = self.compute_precision_at_k(100, target, predictions_proba)
        metrics['precision_at_400'] = self.compute_precision_at_k(400, target, predictions_proba)
        metrics['precision_at_1000'] = self.compute_precision_at_k(1000, target, predictions_proba)
        metrics['precision_at_max'] = self.compute_precision_at_k(len(target), target, predictions_proba)
        if flag_average_precision:
            metrics['average_precision_at_100'] = self.compute_average_precision_at_k(100, target, predictions_proba)
            metrics['average_precision_at_400'] = self.compute_average_precision_at_k(400, target, predictions_proba)
            metrics['average_precision_at_1000'] = self.compute_average_precision_at_k(1000, target, predictions_proba)
            metrics['average_precision'] = self.compute_average_precision(target, predictions_proba)
        return metrics

    def compute_metrics_of_confusion_matrix(self, target, predictions):
        metrics = self.compute_binary_confusion_matrix_values(target, predictions)
        metrics['sensitivity'] = self.compute_sensitivity(metrics['TP'], metrics['FN'])
        metrics['specificity'] = self.compute_specificity(metrics['TN'], metrics['FP'])
        metrics['precision'] = precision_score(target, predictions)
        metrics['recall'] = recall_score(target, predictions)
        metrics['f1'] = f1_score(target, predictions)
        return metrics

    def compute_binary_confusion_matrix_values(self, target, predictions):
        metrics = {
            'TP': np.sum(np.logical_and(predictions == 1, target == 1)),
            'TN': np.sum(np.logical_and(predictions == 0, target == 0)),
            'FP': np.sum(np.logical_and(predictions == 1, target == 0)),
            'FN': np.sum(np.logical_and(predictions == 0, target == 1)),
        }
        return metrics

    def compute_roc_auc_score(self, target, predictions_proba):
        if len(set(target)) == 1:
            return np.nan
        return roc_auc_score(target, predictions_proba)

    def compute_sensitivity(self, true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)

    def compute_specificity(self, true_negatives, false_positives):
        return true_negatives / (true_negatives + false_positives)

    def compute_average_precision_at_k(self, k, targets, predictions_proba):
        ranking = self.predictions_to_ranking(predictions_proba)
        average_precision_calculator = AveragePrecisionAtKCalculator(k=k)
        return average_precision_calculator.evaluate(targets, ranking)[f'average_precision_at_{k}']

    def compute_average_precision(self, targets, predictions_proba):
        return average_precision_score(targets, predictions_proba)

    def compute_precision_at_k(self, k, targets, predictions_proba):
        ranking = self.predictions_to_ranking(predictions_proba)
        precision_calculator = PrecisionAtKCalculator(k=k)
        return precision_calculator.evaluate(targets, ranking)[f'precision_at_{k}']

    def predictions_to_ranking(self, predictions):
        order = predictions.argsort()[::-1]
        ranking = order.argsort()
        return ranking
