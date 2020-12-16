# -*- coding: utf-8 -*-
"""

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    *
    *
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import hotel_reviews_preprocessing

EPOCHS = 15

bayes_features, bayes_classes = hotel_reviews_preprocessing.get_features_and_classes('bayes')
pattern_features, pattern_classes = hotel_reviews_preprocessing.get_features_and_classes()


def test_k_models(epochs, features, classes):
    """

    @param epochs:
    @param features:
    @param classes:
    @return:
    """
    model_scores = {}
    for k in range(1, epochs):
        k_model = KNeighborsClassifier(n_neighbors=k)
        model_scores[str(k)] = (cross_val_score(estimator=k_model, X=features, y=classes, cv=4))
    return model_scores


bayes_scores = test_k_models(EPOCHS, bayes_features, bayes_classes)
pattern_scores = test_k_models(EPOCHS, pattern_features, pattern_classes)


def get_best_k(scores):
    """

    @param scores:
    @return:
    """
    vals = [s.mean() for s in scores.values()]
    keys = list(scores.keys())
    return int(keys[vals.index(max(vals))])


bayes_best = get_best_k(bayes_scores)
pattern_best = get_best_k(pattern_scores)
