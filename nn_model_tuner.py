# -*- coding: utf-8 -*-
"""
Module that uses the Keras tuner to tune neural network models (for both pattern and bayes sentiment analysis,
respectively) to find the best hyperparameters to use to construct the most accurate neural network model.

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    * https://www.tensorflow.org/tutorials/keras/keras_tuner
"""
import time
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
import tensorflow as tf
import hotel_reviews_preprocessing as hrp

# Tensorflow setting to utilize GPU 0 to run models
PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

# Used as the file name for storing the Keras tuner results
TEMP_DIR = f"{int(time.time())}"

pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = hrp.get_clean_pattern_data(nn_model=True)
bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = hrp.get_clean_bayes_data(nn_model=True)


def pattern_model_builder(hyperparameters):
    """
    Build a Keras sequential neural network for classifying the pattern sentiment analysis data and hotel ratings.
    @param hyperparameters: dictionary of hyperparameters to refined and used for building the models layers.
    @return: the built network (model)
    """
    seq_model = keras.Sequential()
    nodes = hyperparameters.Int('nodes', min_value=4000, max_value=6000, step=100)
    dropout_rate = hyperparameters.Float('dropout_rate', 0.18, 0.37, 0.01)
    learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    activation_choice = hyperparameters.Choice('activation_funcs', values=['sigmoid', 'tanh', 'relu'])

    seq_model.add(layers.BatchNormalization(input_shape=[2]))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(5, activation='softmax'))

    seq_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

    return seq_model


pattern_tuner = kt.Hyperband(
    pattern_model_builder,
    objective='categorical_accuracy',
    max_epochs=100,
    factor=10,
    directory=TEMP_DIR,
    project_name=TEMP_DIR
)


def bayes_model_builder(hyperparameters):
    """
    Build a Keras sequential neural network for classifying the bayes sentiment analysis data and hotel ratings.
    @param hyperparameters: dictionary of hyperparameters to refined and used for building the models layers.
    @return: the built network (model)
    """
    seq_model = keras.Sequential()
    nodes = hyperparameters.Int('nodes', min_value=4000, max_value=6000, step=200)
    dropout_rate = hyperparameters.Float('dropout_rate', 0.18, 0.37, 0.01)
    learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    activation_choice = hyperparameters.Choice('activation_funcs', values=['sigmoid', 'tanh', 'relu'])

    seq_model.add(layers.BatchNormalization(input_shape=[3]))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(nodes, activation=activation_choice))
    seq_model.add(layers.BatchNormalization())
    seq_model.add(layers.Dropout(dropout_rate))
    seq_model.add(layers.Dense(5, activation='softmax'))

    seq_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

    return seq_model


bayes_tuner = kt.Hyperband(
    bayes_model_builder,
    objective='categorical_accuracy',
    max_epochs=10,
    factor=2,
    directory=TEMP_DIR,
    project_name=TEMP_DIR
)


def tune_pattern_model():
    """
    Used to run the tuner on the neural network pattern sentiment analysis model.

    @return: dict containing the best hyperparameters found during the tuning session
    """
    pattern_tuner.search(
        pattern_x_train, pattern_y_train,
        validation_data=(pattern_x_test, pattern_y_test),
        batch_size=1024,
        epochs=3
    )

    return pattern_tuner.get_best_hyperparameters(num_trials=1)[0]


def tune_bayes_model():
    """
    Used to run the tuner on the neural network pattern bayes analysis model.

    @return: dict containing the best hyperparameters found during the tuning session
    """
    bayes_tuner.search(
        bayes_x_train, bayes_y_train,
        validation_data=(bayes_x_test, bayes_y_test),
        batch_size=512,
        epochs=3
    )

    return bayes_tuner.get_best_hyperparameters(num_trials=1)[0]


best_param_pattern = tune_pattern_model()
best_param_pattern['input_size'] = 2

best_param_bayes = tune_bayes_model()
best_param_bayes['input_size'] = 3

# Parameters that were ultimately used
bayes_parameters = {'nodes': 1550, 'dropout_value': 0.21, 'learning_rate': 0.001, 'epochs': 20, 'input_size': 3}
pattern_parameters = {'nodes': 1550, 'dropout_value': 0.19, 'learning_rate': 0.001, 'epochs': 20, 'input_size': 2}
