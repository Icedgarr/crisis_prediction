"""Base class for all predictors"""

# (c) Telefónica Innovación Alpha. All rights reserved

from abc import ABCMeta, abstractmethod
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy import array
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold

METRIC_WEIGHT = .6


class Predictor(metaclass=ABCMeta):
    def __init__(self, hyperparameters, id_kwargs=None):
        self.hyperparameters = hyperparameters
        self.id_kwargs = id_kwargs or {}

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @staticmethod
    def apply_schemata(func):
        def new_func(self, features=None, target=None):
            kwargs = {}
            if features is not None:
                kwargs['features'] = features[[*self.feature_schema.keys()]]

            if target is not None:
                kwargs['target'] = target[[*self.target_schema.keys()]]

            return func(self, **kwargs)

        return new_func

    @property
    @abstractmethod
    def feature_schema(self):
        pass

    @property
    @abstractmethod
    def target_schema(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def train(self, data, target):
        pass

    @abstractmethod
    def validate(self, data, target):
        pass

    def validate_binary_model(self, features, target, splitter):
        metrics, predictions, targets = [], [], []
        for tr_i, t_i in splitter.split(features, target.iloc[:, 0]):
            train_data = features.iloc[tr_i], target.iloc[tr_i].squeeze()
            test_features, test_target = features.iloc[t_i], target.iloc[t_i].values.reshape(-1)
            model = self.Model(**self.hyperparameters)
            model.fit(*train_data)
            pred_prob = model.predict_proba(test_features)[:, 1]
            metrics.append(self.compute_binary_metrics(test_target, pred_prob))
            predictions.extend(pred_prob.round())
            targets.extend(test_target)

        metrics = pd.DataFrame(metrics).mean().to_dict()
        metrics.update(self.compute_metrics_of_confusion_matrix(np.array(targets), np.array(predictions)))
        return metrics

    def validate_binary_model_stratified(self, features, target, n_splits=5):
        return self.validate_binary_model(features, target, StratifiedKFold(n_splits=n_splits))

    def validate_binary_model_time_series(self, features, target, n_splits=5):
        return self.validate_binary_model(features, target, TimeSeriesSplit(n_splits=n_splits))

    def validate_binary_model_by_user(self, features, target, n_splits=5):
        return self.validate_binary_model(features, target, GroupKFold(n_splits=n_splits))

    def compute_binary_metrics(self, target, predictions_proba):
        metrics = self.compute_metrics_of_confusion_matrix(target, predictions_proba.round())
        metrics['roc_auc'] = self.compute_roc_auc_score(target, predictions_proba)
        metrics['kappa'] = cohen_kappa_score(target, predictions_proba.round())
        return metrics

    def compute_metrics_of_confusion_matrix(self, target, predictions):
        metrics = self.compute_binary_confusion_matrix_values(target, predictions)
        metrics['sensitivity'] = self.compute_sensitivity(metrics['TP'], metrics['FN'])
        metrics['specificity'] = self.compute_specificity(metrics['TN'], metrics['FP'])
        metrics['neutral'] = np.mean([metrics['sensitivity'], metrics['specificity']])
        metrics['high_sensitivity'] = ((metrics['sensitivity'] * METRIC_WEIGHT) +
                                       ((1 - METRIC_WEIGHT) * metrics['specificity']))
        metrics['high_specificity'] = ((metrics['sensitivity'] * (1 - METRIC_WEIGHT)) +
                                       (METRIC_WEIGHT * metrics['specificity']))
        return metrics

    @staticmethod
    def compute_binary_confusion_matrix_values(target, predictions):
        metrics = {
            'TP': np.sum(np.logical_and(predictions == 1, target == 1)),
            'TN': np.sum(np.logical_and(predictions == 0, target == 0)),
            'FP': np.sum(np.logical_and(predictions == 1, target == 0)),
            'FN': np.sum(np.logical_and(predictions == 0, target == 1)),
        }
        return metrics

    @staticmethod
    def compute_roc_auc_score(target, predictions_proba):
        return roc_auc_score(target, predictions_proba)

    @staticmethod
    def compute_sensitivity(true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def compute_specificity(true_negatives, false_positives):
        return true_negatives / (true_negatives + false_positives)


class AlphaModel(metaclass=ABCMeta):
    def __init__(self, feature_names: List[str], target_name: str, hyperparameters: Dict = None,
                 fit_params: Dict = None):
        self.feature_names = feature_names
        self.target_name = target_name
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.fit_params = fit_params if fit_params is not None else {}

    @abstractmethod
    def fit(self, train_data: DataFrame):
        pass

    @abstractmethod
    def predict(self, test_data: DataFrame) -> array:
        pass
