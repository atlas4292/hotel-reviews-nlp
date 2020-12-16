# -*- coding: utf-8 -*-
"""
Module that employs the use of GridSearchCV to tune a K-Nearest Neighbor classification model to find the best value
to use as the k input to the model.

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import hotel_reviews_preprocessing as hrp

EPOCHS = 500

k_parameters = {'n_neighbors': list(range(1, EPOCHS + 1))}
bayes_features, bayes_classes = hrp.get_features_and_classes(bayes=True)
pattern_features, pattern_classes = hrp.get_features_and_classes()


def test_k_models(param_dict, features, classes, cross_val=4):
    """
    Test the model using the parameters (and their ranges) specified in the param_dict input. Features and classes are
    used to fit the model for testing the accuracy score. Cross validation is used for splitting the data throughout
    the grid search.

    @param param_dict: (MUST be) a dict containing the parameters to use for tuning a KNN model
    @param features: the features to train the model with
    @param classes: the classes to train the model with
    @param cross_val: the desired training and testing split value for cross validation
    @return: the best k value found
    """
    assert type(param_dict) == dict
    model = GridSearchCV(KNeighborsClassifier(), param_dict, cv=cross_val)
    model.fit(features, classes)
    return list(model.best_params_.values())[0]


bayes_best = test_k_models(k_parameters, bayes_features, bayes_classes)
pattern_best = test_k_models(k_parameters, pattern_features, pattern_classes)
