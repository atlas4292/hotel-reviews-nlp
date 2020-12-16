# -*- coding: utf-8 -*-
"""
Module contains a variety of tools used for natural language processing on dataset's of text (sentences).
The tools aim to create an accurate numerical representation of the underlying message being conveyed by a
particular sentence.

@author: nferry@email.sc.edu
@version: 1.0

Resources Used:
    * https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
    * https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.en.sentiments.PatternAnalyzer
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelBinarizer
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer


def tokenize_sentences(sentences):
    """
    Tokenize each sentence in a list of sentences into their individual word components. Append the tokenized sentence
    to a new dictionary whose index is based on the index of processing.

    @param sentences: a list whose components are whole sentences
    @return: dict of tokenized sentences
    """
    tokenized_reviews = {}
    for i, sent in enumerate(sentences):
        tokenized_reviews[str(i)] = word_tokenize(sent)
    return tokenized_reviews


def remove_stop_words(sentences):
    """
    Removes stop words and symbols from each individual sentence contained within the supplied input list of sentences.

    @param sentences: list of sentences
    @return: list of sentences with stop words removed from each individual sentence component
    """
    tokenized_sentences = tokenize_sentences(sentences)
    stop_words = set(stopwords.words("english"))
    for i, tokenized_sent in enumerate(tokenized_sentences.values()):
        word_list = []
        for word in tokenized_sent:
            if word not in stop_words:
                word_list.append(word)
        tokenized_sentences[str(i)] = ' '.join(w for w in word_list if w.isalnum())

    return tokenized_sentences.values()


def sentiment_analysis(sentences, analyzer=None):
    """
    Perform sentiment analysis on each individual sentence contained within the list of input sentences. The analyzer
    key arg is used to specify the type of sentiment analyzer to use (default analyzer: pattern analysis)

    @param sentences: list of sentences
    @param analyzer: the type of analyzer desired to be used to perform the sentiment analysis
    @return: list of sentiment analysis for every sentence from 'sentences'
    """
    return [TextBlob(sentence, analyzer=analyzer).sentiment for sentence in sentences]


def split_pattern_sentiments(sentiments):
    """
    Separate the individual pattern sentiment analysis components for every sentiment analysis
    within the list of input sentiments.

    @param sentiments: list of sentiment analyses
    @return: the separated sentiment analysis components; each component is a list
    """
    polarities = [sentiment[0] for sentiment in sentiments]
    subjectivities = [sentiment[1] for sentiment in sentiments]

    return polarities, subjectivities


def split_naivebayes_sentiments(sentiments):
    """
    Separate the individual bayes sentiment analysis components for every sentiment analysis
    within the list of input sentiments.

    @param sentiments: list of sentiment analyses
    @return: the separated sentiment analysis components; each component is a list
    """
    polar_class = [sentiment[0] for sentiment in sentiments]
    pct_pos = [sentiment[1] for sentiment in sentiments]
    pct_neg = [sentiment[2] for sentiment in sentiments]

    return polar_class, pct_pos, pct_neg


def pattern_sentiment_analysis(sentences_df):
    """
    Perform the pattern sentiment analysis on the input dataset and return the analysis,
    separated into it's individual components, as a data frame.

    @param sentences_df: data frame of sentences to perform analysis on
    @return: data frame of the sentiment analysis results, broken down into individual components
    """
    review_sentiments = sentiment_analysis(sentences_df.iloc[:, 0])
    review_polarities, review_subjectivity = split_pattern_sentiments(review_sentiments)
    return pd.DataFrame({'Polarity': review_polarities,
                         'Subjectivity': review_subjectivity,
                         'Rating': sentences_df.iloc[:, 1]})


def bayes_sentiment_analysis(sentences_df):
    """
    Perform the bayes sentiment analysis on the input dataset and return the analysis,
    separated into it's individual components, as a data frame.

    @param sentences_df: data frame of sentences to perform analysis on
    @return: data frame of the sentiment analysis results, broken down into individual components
    """
    review_naivebayes = sentiment_analysis(sentences_df.iloc[:, 0], analyzer=NaiveBayesAnalyzer())
    polar_class, pct_neg, pct_pos = split_naivebayes_sentiments(review_naivebayes)
    df = pd.DataFrame({'Classification': polar_class,
                       'Percent Negative': pct_neg,
                       'Percent Positive': pct_pos,
                       'Rating': sentences_df.iloc[:, 1]})
    return df.assign(Classification=label_binize(df.iloc[:, 0]))


def label_binize(dataset):
    """
    Binary (re)labels a dataset's binary categories with 1 and 0.

    @param dataset: binary categorical data to binary label
    @return: binary labeled dataset
    """
    return LabelBinarizer().fit_transform(dataset)
