import numpy as np
from collections import Counter
import spacy

print("loading spacy medium model")
nlp = spacy.load("en_core_web_md")
print("finished loading spacy medium model")


def get_features_multi(articles, similarity_matrix):
    """
    :param articles: [[[sentences]article]cluster]
    :param similarity_matrix [[i,j is similarity of sentence i,j]]
    :return:
    """
    num_features = 7
    total_sentences = sum(len(article) for article in articles)
    feature = np.zeros((num_features, total_sentences))

    # increments for each article and continues to increment from one article to the next to fill in the feature matrix
    j = 0
    for article in articles:
        for i, sentence in enumerate(article):
            #need to calculate the number of sentences in the article leading up to a sentence
            sentence_string = " ".join(sentence)
            num_sentence_words = len(sentence)
            feature[0, j] = num_sentence_words  # num words
            feature[1, j] = len(sentence_string)  # num chars
            feature[2, j] = i  # position in article
            feature[3, j] = (np.sum(similarity_matrix[i]) - similarity_matrix[i][i]) / (len(similarity_matrix) - 1)        #mean cluster similarity
            feature[4, j] = sum(word[0].isdigit() for word in sentence) / num_sentence_words

            nlp_sentence = nlp(sentence_string)
            num_noun_phrases = len(list(nlp_sentence.noun_chunks))
            num_verbs = sum(1 for token in nlp_sentence if token.pos_ == "VERB")
            feature[5, j] = num_noun_phrases / num_sentence_words
            feature[6, j] = num_verbs / num_sentence_words
            j += 1

    return feature


def get_S_multi(cluster_sentences_X, inv_doc_freq):
    """
    :param cluster_sentences_X:  [[sentences]cluster]
    :param inv_doc_freq: dictionary {'name1': value1, 'name2': value2}
    :return:
    """

    missing_inv_freq = -np.log(1 / 10_000)  # assume unseen words appear once every 10k articles

    # pre compute sets, counters for efficiency
    sets = [set(sentence) for sentence in cluster_sentences_X]
    counters = [Counter(sentence) for sentence in cluster_sentences_X]

    total_sentences = len(cluster_sentences_X)
    S = np.zeros((total_sentences, total_sentences))
    for i in range(total_sentences):
        for j in range(total_sentences):
            if i > j:
                #S symmetric - efficiency
                S[i, j] = S[j, i]
                continue
            set_i = sets[i]
            set_j = sets[j]
            counter_i = counters[i]
            counter_j = counters[j]
            common_words = set_i & set_j

            numerator = 0
            for word in common_words:
                numerator += counter_i[word] * counter_j[word] * inv_doc_freq.get(word, missing_inv_freq) ** 2

            denom_i = 0
            for word, count in counter_i.items():
                denom_i += (counter_i[word] * inv_doc_freq.get(word, missing_inv_freq)) ** 2

            denom_j = 0
            for word, count in counter_j.items():
                denom_j += (counter_j[word] * inv_doc_freq.get(word, missing_inv_freq)) ** 2

            S[i, j] = numerator / (denom_i * denom_j) ** 0.5
    return S


def article_cluster_to_words_in_article_cluster(article_sentences_X):
    """
    :param article_sentences_X: [[[sentences]article]cluster]
    :return: [{words in article}cluster]
    """
    words_in_article_cluster = []
    for article in article_sentences_X:
        words_in_article = set()
        for sentence in article:
            words_in_sentence = set(sentence)
            words_in_article = words_in_article.union(words_in_sentence)
        words_in_article_cluster.append(words_in_article)
    return words_in_article_cluster


def compute_and_save_features(datapoints, similarity_matrices):
    features = []  # will take form [[features of cluster]training data]
    print("started calculating features")
    for i, datapoint in enumerate(datapoints):
        _, articles, _, _, _ = datapoint
        # article sentences are [[[sentences]article]cluster]
        feature = get_features_multi(articles, similarity_matrices[i])
        features.append(feature)
    print("finished calculating features")
    [np.save(f'train_na=4_saved_features_{i}.npy', features[i]) for i in range(len(features))]
    #saves features to file so that they need not be recomputed
    return features

def compute_and_save_similarity(datapoints, inv_doc_freq_dict):
    similarity_matrices = []  # will take form [[similarity of cluster]training data]
    print("started calculating similarity")
    for datapoint in datapoints:
        _, _, cluster_sentences, _, _ = datapoint
        # article sentences are [[[sentences]article]cluster]
        similarity = get_S_multi(cluster_sentences, inv_doc_freq_dict)
        similarity_matrices.append(similarity)
    print("finished calculating similarity")
    [np.save(f'train_na=4_saved_similarities_{i}.npy', similarity_matrices[i]) for i in range(len(similarity_matrices))]
    #saves similarity to file so need not recompute
    return similarity_matrices