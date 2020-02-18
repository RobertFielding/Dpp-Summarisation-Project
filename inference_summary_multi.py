from operator import itemgetter
from sentence_features_multi import *


def predict(theta, features, S, cluster_sentences, num_highlights):
    num_sentences_cluster = len(cluster_sentences)

    U = list(range(num_sentences_cluster))  # list of indexes of sentences of X
    indices_Y = []

    while len(indices_Y) < num_highlights and len(U) > 0:
        contributions = []  # we will add the sentence that contributes most to making a good summary

        for i in U:
            # what is the contribution of adding i to the existing set of indices_Y
            contribution = np.exp(np.matmul(np.transpose(theta), features[:, i])) * submatrix_det(S, indices_Y + [i])
            contributions.append((i, contribution))
        idx_max = max(contributions, key=itemgetter(1))[0]
        # gets index of maximum contribution
        indices_Y.append(idx_max)
        U.remove(idx_max)

    return [cluster_sentences[index] for index in indices_Y]


def submatrix_det(matrix, indices):
    return np.linalg.det(matrix[np.ix_(indices, indices)])