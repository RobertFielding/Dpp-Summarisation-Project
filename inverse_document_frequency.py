import json
from collections import defaultdict
import numpy as np


def articles_to_words_in_articles(articles):
    """
    :param articles: [[[sentences as words]article]training data]
    :return: [{words in article}training dataset]
    """
    words_in_articles = []  # will take form [{article words}training data]
    for article in articles:
        sentence_words = [set(sentence) for sentence in article]
        if len(sentence_words) > 0:
            article_words = set.union(*sentence_words)  # * unpacks list into multiple arguments
            words_in_articles.append(article_words)
    return words_in_articles


def inv_doc_freq(articles):
    """
    :param articles: [[[sentences as list of words]article]training set]
    :return: dictionary
    """
    words_in_articles = articles_to_words_in_articles(articles)  # [{words in article}training dataset]
    num_articles = len(articles)
    word_frequency = defaultdict(int) #dictionary where entries default to 0

    for article in words_in_articles:
        for word in article:
            word_frequency[word] += 1


    inverse_frequencies = {}
    for word, frequency in word_frequency.items(): #chooses word and frequency from default dict
        inverse_frequencies[word] = - np.log(frequency / num_articles)
    return inverse_frequencies


def compute_and_save(datapoints, output_file):
    all_articles = []  # will take form [[[sentences as words]article]training data]
    for datapoint in datapoints:
        _, articles, _, _, _ = datapoint
        # article sentences are [[[sentences]article]cluster]
        for article in articles:
            all_articles.append(article)

    print("started processing word counts")
    inv_doc_freq_dict = inv_doc_freq(all_articles)
    print("finished processing word counts")
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(inv_doc_freq_dict, f, separators=(',', ':'))
        #stores dictionary in JSON file named output_file
    return inv_doc_freq_dict
