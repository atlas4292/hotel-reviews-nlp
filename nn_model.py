# -*- coding: utf-8 -*-
"""
Module that builds the best neural network softmax classification model for both the bayes and pattern sentiment
analysis datasets, based on tuned hyperparameters found with the nn_model_tuner module.

@author: nferry@email.sc.edu
@version: 1.0
"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import hotel_reviews_preprocessing as hrp
from nn_model_tuner import bayes_parameters, pattern_parameters

# Tensorflow setting to utilize GPU 0 to run models
PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

BATCH_SIZE = 1024

pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = hrp.get_clean_pattern_data(nn_model=True)
bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = hrp.get_clean_bayes_data(nn_model=True)


def build_model(hyperparameters):
    """
    Build and compile a neural network model whose hyperparameters are supplied by the input dict
    @param hyperparameters: dict of hyperparameters to use in the construction of the neural network
    @return: the built and compiled neural network model
    """
    model = keras.Sequential()

    model.add(layers.BatchNormalization(input_shape=[hyperparameters['input_size']]))
    model.add(layers.Dense(hyperparameters['nodes'], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparameters['dropout_value']))
    model.add(layers.Dense(hyperparameters['nodes'], activation='sigmoid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparameters['dropout_value']))
    model.add(layers.Dense(hyperparameters['nodes'], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparameters['dropout_value']))
    model.add(layers.Dense(hyperparameters['nodes'], activation='sigmoid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hyperparameters['dropout_value']))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


bayes_model = build_model(bayes_parameters).fit(bayes_x_train, bayes_x_test,
                                                validation_data=(bayes_y_train, bayes_y_test),
                                                epochs=bayes_parameters['epochs'],
                                                batch_size=BATCH_SIZE)

pattern_model = build_model(pattern_parameters).fit(pattern_x_train, pattern_x_test,
                                                    validation_data=(pattern_y_train, pattern_y_test),
                                                    epochs=pattern_parameters['epochs'],
                                                    batch_size=BATCH_SIZE)
