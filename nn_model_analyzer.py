# -*- coding: utf-8 -*-
"""
Module that aims to analyze a compiled and fitted neural network model. The imported model analysis toolkit provides
additional model analysis functionality.

@author: nferry@email.sc.edu
@version: 1.0
"""
from tensorflow.keras.utils import plot_model
import hotel_reviews_preprocessing as hrp
import model_analysis_toolkit as mat
import nn_model

pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = hrp.get_clean_pattern_data(nn_model=True)
bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = hrp.get_clean_bayes_data(nn_model=True)

bayes_nn_model = nn_model.bayes_model
pattern_nn_model = nn_model.pattern_model

bayes_predict = bayes_nn_model.predict(bayes_x_test)
pattern_predict = pattern_nn_model.predict(pattern_x_test)

bayes_confusion_matrices = mat.create_confusion_matrices(bayes_y_test, bayes_predict, sparse=True)
pattern_confusion_matrices = mat.create_confusion_matrices(pattern_y_test, pattern_predict, sparse=True)

bayes_classification_report = mat.create_classification_report(bayes_y_test, bayes_predict, sparse=True)
pattern_classification_report = mat.create_classification_report(pattern_y_test, pattern_predict, sparse=True)


def plot_nn_model(compiled_model, file_name=''):
    """
    Function that creates an image representation of the neural network. The image is saved and named based on the
    supplied file name.

    @param compiled_model: compiled neural network model
    @param file_name: string representing the desired file name for saving the plot
    """
    plot_model(compiled_model, to_file='./output/figures/models' + file_name + '_nn_model.png', show_shapes=True)
