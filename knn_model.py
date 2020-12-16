# -*- coding: utf-8 -*-
"""
Module that builds the best K-Nearest Neighbor classification model for both the bayes and pattern sentiment analysis
datasets, based on tuned hyperparameters found with the knn_model_tuner_v2 module.


@author: nferry@email.sc.edu
@version: 1.0
"""
from sklearn.neighbors import KNeighborsClassifier

import hotel_reviews_preprocessing as hrp
from knn_model_tuner_v2 import bayes_best, pattern_best

bayes_x_train, bayes_x_test, bayes_y_train, bayes_y_test = hrp.get_clean_bayes_data()
pattern_x_train, pattern_x_test, pattern_y_train, pattern_y_test = hrp.get_clean_pattern_data()

bayes_model = KNeighborsClassifier(n_jobs=bayes_best).fit(bayes_x_train, bayes_y_train)
pattern_model = KNeighborsClassifier(n_jobs=pattern_best).fit(pattern_x_train, pattern_y_train)
