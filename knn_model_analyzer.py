# -*- coding: utf-8 -*-
"""
Module that aims to analyze a compiled and fitted K-Nearest Neighbor model. The imported model analysis toolkit provides
additional model analysis functionality.

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    * http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mlxtend.plotting import plot_decision_regions

import hotel_reviews_preprocessing as hrp
import knn_model
import model_analysis_toolkit as mat

pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = hrp.get_clean_pattern_data()
bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = hrp.get_clean_bayes_data()

bayes_knn_model = knn_model.bayes_model
pattern_knn_model = knn_model.pattern_model

bayes_predict = bayes_knn_model.predict(bayes_x_test)
pattern_predict = pattern_knn_model.predict(pattern_x_test)

# Create confusion matrices for each classification label (i.e., rating 1-5)
bayes_confusion_matrices = mat.create_confusion_matrices(bayes_y_test, bayes_predict)
pattern_confusion_matrices = mat.create_confusion_matrices(pattern_y_test, pattern_predict)

bayes_classification_report = mat.create_classification_report(bayes_y_test, bayes_predict)
pattern_classification_report = mat.create_classification_report(pattern_y_test, pattern_predict)


def plot_knn_model(knn_model, x_test, y_test, feature_fill_values, file_name='', bayes=False):
    """
    Used to plot the Knn models decision boundaries alongside the feature and class testing data. Saves the model
    with and if a file_name is supplied. If there are more than two features, the additional features values will
    have to be specified as filler_value with a +/- margin of filler_range.


    @param knn_model: knn model to use for plotting
    @param x_test: feature test data to use
    @param y_test: class test data to use
    @param feature_fill_values: list expected [filler_index, filler_value, filler_range]
    @param file_name: file name to save the plot as
    @param bayes: bool indicator to fill the third feature necessary for bayes model
    """
    assert type(feature_fill_values) == list
    filler_index, filler_value, filler_range = feature_fill_values
    gs = gridspec.GridSpec(5, 2)
    fig = plt.figure(figsize=(10, 8))
    if bayes:
        fig = plot_decision_regions(X=x_test, y=y_test, clf=knn_model, legend=5,
                                    filler_feature_values={filler_index: filler_value},
                                    filler_feature_ranges={filler_index: filler_range})
        x_label = 'Sentence % Negative'
        y_label = 'Sentence % Positive'
        title = 'Bayes'
    else:
        fig = plot_decision_regions(X=x_test, y=y_test, clf=knn_model, legend=5)
        x_label = 'Sentence Polarity'
        y_label = 'Sentence Subjectivity'
        title = 'Pattern'

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + ' KNN Model')
    if file_name:
        plt.savefig('./output/figures/models/' + file_name + '_knn_model.png')
    plt.show()
