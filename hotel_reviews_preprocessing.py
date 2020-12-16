# -*- coding: utf-8 -*-
"""
Module for wrangling/preprocessing the hotel reviews and ratings dataset so that the dataset can be used in an ML
classification model.

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    * https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
    * https://www.kaggle.com/datafiniti/hotel-reviews
"""
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import preprocessing_nlp as nlp

hotel_reviews = pd.read_excel('./input/complete_hotel_reviews.xlsx')
hotel_reviews.iloc[:, 0] = nlp.remove_stop_words(hotel_reviews.iloc[:, 0])


def split_train_test(features, classes, test_size=0.1):
    """
    Split the feature corresponding class data into training and testing partitions. The split percentage is based on
    the provided test size.

    @param features: the feature set of the data
    @param classes: the class labels of the data
    @param test_size: the split percentage of data partitioning
    @return: training and testing sets for x (features) and y (classes)
    """
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        classes,
                                                        test_size=test_size,
                                                        stratify=classes,
                                                        random_state=int(random.random() * 100))

    return x_train, x_test, y_train, y_test


def one_hot_encoder(dataset):
    """
    One hot encode the input dataset.
    @param dataset: set of tabular data to be encoded
    @return: one hot encoded tabular data
    """
    return OneHotEncoder().fit_transform(dataset)


def get_clean_pattern_data(nn_model=False):
    """
    Acquire the wrangled and pattern sentiment natural-language processed hotel reviews data. Data is one hot encoded if
    the model being used is a neural network and is specified by marking nn_model as True. Default is KNN model.

    @param nn_model: activation key for one hot encoding the data if the model is a neural network
    @return: Pattern sentiment classified data wrangled and split into x and y (training and testing) data
    """
    pattern_df = nlp.pattern_sentiment_analysis(hotel_reviews)
    shape = [2] if nn_model else 2
    pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = \
        train_test_split(pattern_df.iloc[:, 0:2], pattern_df.iloc[:, shape], test_size=0.2)
    if nn_model:
        pattern_y_train = pd.DataFrame.sparse.from_spmatrix(one_hot_encoder(pattern_y_train))
        pattern_y_test = pd.DataFrame.sparse.from_spmatrix(one_hot_encoder(pattern_y_test))

    return pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test


def get_clean_bayes_data(nn_model=False):
    """
    Acquire the wrangled and bayes sentiment natural-language processed hotel reviews data. Data is one hot encoded if
    the model being used is a neural network and is specified by marking nn_model as True. Default is KNN model.

    @param nn_model: activation key for one hot encoding the data if the model is a neural network
    @return: Bayes sentiment classified data wrangled and split into x and y (training and testing) data
    """
    bayes_df = nlp.bayes_sentiment_analysis(hotel_reviews)
    shape = [3] if nn_model else 3
    bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = \
        train_test_split(bayes_df.iloc[:, 0:3], bayes_df.iloc[:, shape], test_size=0.2)
    if nn_model:
        bayes_y_train = pd.DataFrame.sparse.from_spmatrix(one_hot_encoder(bayes_y_train))
        bayes_y_test = pd.DataFrame.sparse.from_spmatrix(one_hot_encoder(bayes_y_test))

    return bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test


def clean_7282_data():
    """
    Wrangles the 7282_1 dataset of hotel reviews and ratings
    """
    data = pd.read_excel('input/7282_1_dirty.xlsx')

    clean_data = data[data.Reviews.notnull()]
    clean_data = clean_data[clean_data.Rating != 0]
    clean_data = clean_data[clean_data.Rating.notnull()]
    clean_data = clean_data[clean_data.Rating <= 5]
    clean_data.to_excel('output/clean_data/clean_7282_data.xlsx')


def get_features_and_classes(bayes=False):
    """
    Get the features and classes of the bayes or pattern sentiment analyzed (and wrangled) data. Default is pattern
    sentiment analysis data.

    @param bayes: bool check for acquiring bayes sentiment analysis type dataset
    @return: features and classes of the desired sentiment analysis type dataset
    """
    if bayes:
        df = nlp.bayes_sentiment_analysis(hotel_reviews)
        features = df.iloc[:, 0:3]
        classes = df.iloc[:, 3]
    else:
        df = nlp.pattern_sentiment_analysis(hotel_reviews)
        features = df.iloc[:, 0:2]
        classes = df.iloc[:, 2]

    return features, classes
