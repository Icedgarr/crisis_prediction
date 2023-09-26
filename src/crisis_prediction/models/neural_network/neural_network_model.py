import os
import random

import numpy as np
from pandas import DataFrame
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from crisis_prediction.models.base import AlphaModel
from crisis_prediction.models.neural_network.weighted_cross_entropy import WeightedBinaryCrossEntropy

DEFAULT_NEURAL_NETWORK_HYPERPARAMETERS = {
    'output_dim': 1,
    'epochs': 50,
    'with_batch_norm': False,
    'seed_value': 0,
    'activity_regularizer': None,
    'batch_size': 2048,
    'dropout_rate': 0.5,
    'embedding_dim': 768,
    'kernel_regularizer': None,
    'layer_size': 500,
    'learning_rate': 2e05,
    'loss': WeightedBinaryCrossEntropy,
    'num_layers': 6

}


class NeuralNetworkModel(AlphaModel):

    def __init__(self, struct_features_names, target_name, hyperparameters,
                 embedding_feature_name=None, weeks_elapsed_feature_name=None, callbacks=None, num_time_inputs=1):
        super().__init__(struct_features_names, target_name, hyperparameters)
        self.hyperparameters = dict(DEFAULT_NEURAL_NETWORK_HYPERPARAMETERS, **self.hyperparameters)
        self.struct_features = struct_features_names
        self.feature_names = struct_features_names.copy()
        self.embedding_feature = embedding_feature_name
        self.weeks_elapsed_feature = weeks_elapsed_feature_name
        self.callbacks = callbacks
        self.struct_features_only = embedding_feature_name is None
        self.seed_value = self.hyperparameters.get('seed_value')
        self.num_time_inputs = num_time_inputs
        self.layer_size = self.hyperparameters['layer_size']
        self.with_batch_norm = self.hyperparameters.get('with_batch_norm')
        self._build_network()
        self._set_seeds()

    def fit(self, data: DataFrame, retrain=False):
        if not retrain:
            self.model.set_weights(self.initial_weights)
        input_data = self._get_input_data(data)
        self.model.fit(input_data,
                       data[self.target_name],
                       batch_size=self.hyperparameters['batch_size'],
                       epochs=self.hyperparameters['epochs'],
                       verbose=self.hyperparameters['verbosity'],
                       callbacks=self.callbacks)

    def predict(self, data: DataFrame):
        input_data = self._get_input_data(data)
        predictions = self.model.predict(input_data).reshape(-1)
        return predictions

    def _build_network(self):
        dropout_rate = self.hyperparameters['dropout_rate']
        num_layers = self.hyperparameters['num_layers']
        learning_rate = self.hyperparameters['learning_rate']

        struct_input = Input((len(self.struct_features),))
        struct_normalized = BatchNormalization()(struct_input)
        dense_struct1 = Dense(self.layer_size, activation='relu',
                              kernel_initializer=glorot_uniform(seed=self.seed_value))(struct_normalized)
        dropout_dense_struct1 = Dropout(dropout_rate, seed=self.seed_value)(dense_struct1)

        if not self.struct_features_only:
            word_input = Input((self.hyperparameters['embedding_dim'],))
            time_input = Input((self.num_time_inputs,))
            dense_word1 = Dense(self.layer_size, activation='relu',
                                kernel_initializer=glorot_uniform(seed=self.seed_value))(word_input)
            dropout_dense_word1 = Dropout(dropout_rate, seed=self.seed_value)(dense_word1)
            concat_layer = Concatenate()([dropout_dense_word1, time_input, dropout_dense_struct1])
        else:
            concat_layer = dropout_dense_struct1

        dense_layer = {}
        dropout_layer = {}

        dense_layer[0] = self._add_dense_layer(concat_layer)
        dropout_layer[0] = Dropout(dropout_rate, seed=self.seed_value)(dense_layer[0])

        for i in range(1, num_layers):
            dense_layer[i] = self._add_dense_layer(dropout_layer[i - 1])
            dropout_layer[i] = Dropout(dropout_rate, seed=self.seed_value)(dense_layer[i])

        output = Dense(1, activation='sigmoid',
                       kernel_initializer=glorot_uniform(seed=self.seed_value))(dense_layer[num_layers - 1])
        if self.struct_features_only:
            self.model = Model(inputs=[struct_input], outputs=output)
        else:
            self.model = Model(inputs=[struct_input, word_input, time_input], outputs=output)
            self.feature_names.append(self.embedding_feature)
            self.feature_names.append(self.weeks_elapsed_feature)

        self.model.compile(
            loss=self.hyperparameters['loss'],
            optimizer=Adam(lr=learning_rate),
        )
        self.initial_weights = self.model.get_weights()

    def _add_dense_layer(self, input_layer):
        layer = Dense(self.layer_size * 2, activation='relu',
                      kernel_initializer=glorot_uniform(seed=self.seed_value),
                      kernel_regularizer=self.hyperparameters['kernel_regularizer'],
                      activity_regularizer=self.hyperparameters['activity_regularizer'])(input_layer)
        if self.with_batch_norm:
            layer = BatchNormalization()(layer)
        return layer

    def _get_input_data(self, data):
        if self.struct_features_only:
            input_data = data[self.struct_features]
        else:
            input_data = [data[self.struct_features],
                          np.stack(data[self.embedding_feature]),
                          data[self.weeks_elapsed_feature]]
        return input_data

    def _set_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
