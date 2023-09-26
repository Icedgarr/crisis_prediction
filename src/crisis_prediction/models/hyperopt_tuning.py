from hyperopt import fmin, tpe, STATUS_OK, Trials
import collections
from crisis_prediction.validators.validate_time_split import validate_time_split_data


class HyperoptTuneHyperparams:

    def __init__(self, loader, preprocessors, model_class, config, feature_names, target_name,
                 integer_params, model_hyperparameters, optimized_metric='roc_auc', trials=None):
        self.loader = loader
        self.preprocessors = preprocessors
        self.Model = model_class
        self.feature_names = feature_names
        self.target_name = target_name
        self.integer_params = integer_params
        self.config = config
        self.optimized_metric = optimized_metric
        self.trials = trials if trials is not None else Trials()
        self.model_hyperparameters = model_hyperparameters

    def tune_hyperparameters(self, space, num_trials, algorithm=tpe.suggest):

        best_hyperparams = fmin(fn=self.objective_function,
                                space=space,
                                algo=algorithm,
                                max_evals=num_trials,
                                trials=self.trials)
        return best_hyperparams

    def objective_function(self, params):
        '''for key, value in params.items():
            if isinstance(value, dict):
                print(value)
                for param_key, param_value in value.items():
                    params[param_key] = param_value'''
        params = self.flatten(params)

        for param in self.integer_params:
            params[param] = int(params[param])

        preprocessors = self.preprocessors.copy()
        if params.get('preprocessors'):
            preprocessors += params['preprocessors']

        if params.get('train_start'):
            self.config['train_start'] = params['train_start']

        feature_names_used = self.feature_names.copy()
        if params.get('extra_features'):
            feature_names_used += params['extra_features']

        hyperparameters = {param: value for param, value in params.items() if param in self.model_hyperparameters}
        model = self.Model(feature_names_used, self.target_name, hyperparameters)
        train_metrics, test_metrics, train_data, test_data = validate_time_split_data(model, self.loader,
                                                                                      self.config,
                                                                                      preprocessors)
        return {'loss': -test_metrics[self.optimized_metric], 'status': STATUS_OK}

    def flatten(self, d):
        items = []
        for k, v in d.items():
            new_key = k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v).items())
            else:
                items.append((new_key, v))
        return dict(items)
