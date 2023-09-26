from tensorflow.keras.callbacks import Callback
import pandas as pd
import numpy as np

precision_per_epoch = []
precision_per_epoch_train = []


class EpochMetrics(Callback):
    def __init__(self, train_features, train_target, test_features, test_target, struct_features,
                 embedding_feature_name='Embeddinganonymized_text',
                 weeks_elapsed_feature_name='weeks_elapsed_normalized',
                 grouping_feature='year_week',
                 k=200):
        super(Callback, self).__init__()
        self.k = k
        self.train_features = train_features
        self.train_target = train_target
        self.test_features = test_features
        self.test_target = test_target
        self.struct_features = struct_features
        self.embedding_feature_name = embedding_feature_name
        self.weeks_elapsed_feature_name = weeks_elapsed_feature_name
        self.grouping_feature = grouping_feature

    def on_epoch_end(self, batch, logs={}):
        global precision_per_epoch
        global precision_per_epoch_train
        predictions_proba_train = self.model.predict([self.train_features[self.struct_features],
                                                      np.stack(self.train_features[self.embedding_feature_name]),
                                                      np.stack(self.train_features[self.weeks_elapsed_feature_name])],
                                                     verbose=1,
                                                     batch_size=2048)

        target_train_df = pd.concat([pd.Series(self.train_features[self.grouping_feature].values),
                                     pd.Series(self.train_target),
                                     pd.Series(predictions_proba_train.reshape(-1))], axis=1).rename(
            columns={0: self.grouping_feature,
                     1: 'target',
                     2: 'pred_proba'})
        metric_train = target_train_df.groupby(self.grouping_feature).apply(lambda gp: precision_at_k(gp, self.k))
        print(f'Train precision @ {self.k}:', metric_train.mean())
        precision_per_epoch_train.append(metric_train)
        predictions_proba = self.model.predict([self.test_features[self.struct_features],
                                                np.stack(self.test_features[self.embedding_feature_name]),
                                                np.stack(self.test_features[self.weeks_elapsed_feature_name])],
                                               verbose=1,
                                               batch_size=2048)

        target_df = pd.concat([pd.Series(self.test_features[self.grouping_feature].values),
                               pd.Series(self.test_target),
                               pd.Series(predictions_proba.reshape(-1))],
                              axis=1).rename(columns={0: self.grouping_feature,
                                                      1: 'target',
                                                      2: 'pred_proba'})
        metric = target_df.groupby(self.grouping_feature).apply(lambda gp: precision_at_k(gp, self.k))
        print(f'Precision @ {self.k}:', metric.mean())
        precision_per_epoch.append(metric)
        return


def precision_at_k(group, k):
    sorted_group = group.sort_values(by='pred_proba')
    return sorted_group[-k:]['target'].mean()
