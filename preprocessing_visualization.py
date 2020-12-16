# -*- coding: utf-8 -*-
"""
Module contains a variety of plotting tools that can be used yo visualize the data after/during
pre-processing and before model training (fitting).

@author: nferry@email.sc.edu
@version: 1.0
"""
import matplotlib.pyplot as plt
import seaborn as sns
import hotel_reviews_preprocessing as hrp

bayes_features, bayes_classes = hrp.get_features_and_classes('bayes')
pattern_features, pattern_classes = hrp.get_features_and_classes('knn')


def combine_features_and_classes(features, classes):
    """
    Combine the nlp and wrangled features and classes into one dataset.

    @param features: features dataset
    @param classes: corresponding classes dataset
    @return: joined dataset
    """
    return features.copy().insert(features.shape[1], 'Ratings', classes)


bayes_reviews = combine_features_and_classes(bayes_features, bayes_classes)
pattern_reviews = combine_features_and_classes(pattern_features, pattern_classes)


def plot_bar_chart(features, classes, file_name='', title=''):
    """
    Plot a bar chart and save the chart if a file name is supplied.

    @param features: feature dataset
    @param classes:  corresponding class dataset
    @param file_name: file name to give to the chart when saving the chart
    @param title: title to give to the chart
    """
    plt.bar(features, classes, color='navy')
    for i, val in enumerate(reversed(features)):
        plt.text(i + 0.8, val + 2, str(val), color='firebrick', fontweight='bold')
    plt.title(title + ' Counts')
    if file_name:
        plt.savefig('./output/figures/preprocessing/' + file_name + '_bar.png')
    plt.show()


def plot_cluster_map(dataset, file_name='', title=''):
    """
    Plot a cluster map and save the map if a file name is supplied.

    @param dataset: dataset to visualize
    @param file_name: file name to give to the map when saving the map
    @param title: title to give to the map
    """
    sns.clustermap(dataset).fig.suptitle(title + ' Hierarchical Cluster Map')
    if file_name:
        plt.savefig('./output/figures/preprocessing/' + file_name + '_clustermap.png')
    plt.show()


def plot_heat_map(dataset, file_name='', title=''):
    """
    Plot a heat map and save the map if a file name is supplied.

    @param dataset: dataset to visualize
    @param file_name: file name to give to the map when saving the map
    @param title: title to give to the map
    """
    sns.heatmap(dataset.corr())
    plt.title(title + ' Feature Heat Map')
    if file_name:
        plt.savefig('./output/figures/preprocessing/' + file_name + '_heatmap.png')
    plt.show()


def plot_dist(features, file_name='', title=''):
    """
    Plot a distribution plot and save the histogram if a file name is supplied.

    @param features: feature dataset
    @param file_name: file name to give to the distribution plot when saving the distribution plot
    @param title: title to give to the distribution plot
    """
    sns.displot(features)
    plt.title(title + ' Distribution Plot')
    if file_name:
        plt.savefig('./output/figures/preprocessing/' + file_name + '_dist.png')
    plt.show()


def plot_paired_plots(dataset, file_name='', title='', plot_type=None):
    """
    Plot the paired plots of the dataset.

    @param dataset: dataset to visualize
    @param file_name: file name to give to the distribution plot when saving the distribution plot
    @param title: title to give to the distribution plot
    @param plot_type: type of paired plots to display
    """
    sns.pairplot(dataset, kind=plot_type)
    plt.title(title + ' Paired Plots')
    if file_name:
        plt.savefig('./output/figures/preprocessing/' + file_name + '_pairplots.png')
    plt.show()


def plot_pairgrids(dataset, file_name='', title='', plot_type=None):
    """
    TODO: Use sns.jointplot to create the paired grid plots
    @param dataset: dataset to visualize
    @param file_name:
    @param title:
    @param plot_type:
    """
    pass
