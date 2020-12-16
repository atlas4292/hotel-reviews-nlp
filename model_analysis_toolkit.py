# -*- coding: utf-8 -*-
"""
Module containing a variety of tools for model performance analysis.

@author: nferry@email.sc.edu
@version: 1.0
Resources Used:
    * https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
"""
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score


def create_confusion_matrices(y_test, y_predict, sparse=False):
    """
    Create an individual confusion matrix for each classification label in y_test (and y_predict for that matter).

    @param y_test: class test data
    @param y_predict: class data predicted by a model
    @param sparse: bool indicator to reduce sparse matrix to their true label values
    @return: an ndarray containing the individual matrices for each class label
    """
    if sparse:
        matrices = multilabel_confusion_matrix(y_test.values.argmax(axis=1), y_predict.argmax(axis=1))
    else:
        matrices = multilabel_confusion_matrix(y_test.values, y_predict)
    return matrices


def compute_figure_size(n_plots):
    """
    Computes the necessary figure size needed based on the number of plots to allow for enough room to plot
    all desired plots on a figure.

    @param n_plots: number of plots needed to fit on the figure
    @return: the three digit plot size (xxx = n_row/n_col/plot_index)
    """
    col_size = (n_plots // 2) * 10
    row_size = (n_plots // 2) * 100
    plot_size = col_size + row_size
    if n_plots % 2 > 0:
        plot_size += 100
    return plot_size


def create_figure(n_plots, ax_title='Matrix for Val:'):
    """
    Create a sized figure with empty subplots whose number is indicated by n_plots. The title is applied to each
    individual subplot along with the corresponding subplots index.

    @param n_plots: number of plots needed on the figure
    @param ax_title: title to label the individual empty subplots
    @return: the figure
    """
    fig = plt.figure()
    plot_size = compute_figure_size(n_plots)
    ax = [fig.add_subplot(plot_size + 1, title=ax_title + str(1))]
    for i in range(1, n_plots):
        ax.append(fig.add_subplot(plot_size + (i + 1), sharex=ax[0], sharey=ax[0], title=ax_title + str(i + 1)))
    return fig


def plot_confusion_matrices(confusion_matrices, file_name='', fig_title='', tight=False):
    """
    Plots the confusion matrices on one figure where each subplot represents an individual confusion matrix.

    @param confusion_matrices: iterable of matrices
    @param file_name: name to give to the file when saving
    @param fig_title: title to give to the overall figure
    @param tight: bool check to use tight layout or not
    """
    fig = create_figure(len(confusion_matrices))
    for i, matrix in enumerate(confusion_matrices):
        sns.heatmap(matrix, annot=True, cmap='icefire', fmt='g', ax=fig.axes[i], xticklabels=['-', '+'],
                    yticklabels=['-', '+'])
    fig.suptitle(fig_title + ' Confusion Matrices')
    if tight:
        plt.tight_layout()
    if file_name:
        plt.savefig('./output/figures/confusion_matrices/' + file_name + '_confusion_matrices.png')
    plt.show()


def create_classification_report(y_test, y_predict, sparse=False):
    """
    Creates classification reports for a models performances based on the supplied class test data and the
    model predicted class data (predicted from the feature test data).

    @param y_test: the class test data
    @param y_predict: the model predicted class data
    @param sparse: bool indicator to reduce sparse matrix to their true label values
    @return: dict representing the classification report
    """
    if sparse:
        report = classification_report(y_test.values.argmax(axis=1), y_predict.argmax(axis=1), output_dict=True)
    else:
        report = classification_report(y_test.values, y_predict, output_dict=True)
    return report


def save_classification_report(report, file_name=''):
    """
    Save the classification report to a csv file. File name is supplied by file_name.

    @param report: classification report to save
    @param file_name: name to give to the file when saving
    """
    path = './output/classification_reports/' + file_name + '_classification_report.csv'
    pd.DataFrame(report).to_csv(path)


def get_knn_model_accuracy(y_test, y_pred):
    """
    Compute the accuracy score of the class test data and the model predicted class data.

    @param y_test: the class test data
    @param y_pred: the model predicted class data (from the feature test set)
    @return: the accuracy of the model
    """
    return accuracy_score(y_test, y_pred)


def get_nn_model_accuracy(model, x_test, y_test):
    """
    Compute the accuracy score of the supplied nn model on the supplied feature and class test data.

    @param model: neural network model
    @param x_test: feature test data
    @param y_test: class test data
    @return: the accuracy score
    """
    return model.evaluate(x_test, y_test)[1]


def get_auroc_score():
    """
    TODO: Use metrics.roc_auc_score to compute AUROC
    """
    pass


def plot_roc_curves():
    """
    TODO: Use computed AUROC scores to plot the ROC curves using metrics.plot_roc_curve
    """
    pass


def compute_label_loss_and_score():
    """
    TODO: Compute the label loss in lieu of precision with multi-label models. Use label_ranking_average_precision_score
    """
    pass


def singular_example_visual(model, feature):
    """
    TODO:Show how a singular example makes its way through the supplied model
    @param model:
    @param feature:
    @return:
    """
    pass
