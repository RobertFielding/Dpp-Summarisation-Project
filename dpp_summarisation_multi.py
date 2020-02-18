"""this script will use dpp to summarise some text"""

import json
import os
import random
from collections import namedtuple
from scipy.optimize import minimize

import inference_summary_multi
import inverse_document_frequency
from sentence_features_multi import compute_and_save_features
from sentence_features_multi import compute_and_save_similarity
from data_collator_multi import load_datapoints_multi
from likelihood_computer import *


ResultPoint = namedtuple('ResultPoint', 'file_name extractive_highlights predicted_highlights')
num_datapoints = 650
print("started loading data")
datapoints = load_datapoints_multi(num_datapoints)
print("finished loading data")
feature_matrices = []
similarity_matrices = []
feature_Y_sums = []


inv_doc_freq_file = r'train_na=5_inv_doc_freq.json'

#either calculates inverse_document_frequency or opens stored file
print("start inv_doc")
if not os.path.isfile(inv_doc_freq_file):
    inverse_document_frequency_dict = inverse_document_frequency.compute_and_save(datapoints, inv_doc_freq_file)
else:
    with open(inv_doc_freq_file) as f:
        inverse_document_frequency_dict = json.load(f)
print("finish inv_doc")


##Run one of EITHER/OR below. If not previously computed or num_datapoints changes run EITHER, else OR
##EITHER
similarity_matrices = compute_and_save_similarity(datapoints, inverse_document_frequency_dict)
feature_matrices = compute_and_save_features(datapoints, similarity_matrices)
##OR
# print("loading features")
# feature_matrices = [np.load(f"train_na=5_saved_features_{i}.npy") for i in range(num_datapoints)]
# print("finished loading features")
# print("loading similarity")
# similarity_matrices = [np.load(f"train_na=5_saved_similarities_{i}.npy") for i in range(num_datapoints)]
# print("finished loading similarity")


## If features/similarities aren't stored as files (NOT RECOMMENDED)
# print("calculating feature/similarity matrices")
# for datapoint in datapoints[0: num_datapoints]:
#     _, article_sentences, cluster_sentences, _, _ = datapoint
#     features = get_features_multi(article_sentences)
#     feature_matrices.append(features)
#     similarity_matrices.append(get_S_multi(cluster_sentences, inverse_document_frequency_dict))
# print("finished calculating feature/similarity matrices")


#
for i, datapoint in enumerate(datapoints):
    _, articles, cluster_sentences, _, extractive_highlights = datapoint # _ ignores non-relevant data
    features = feature_matrices[i]

    indices_of_highlight_sentence = set()
    for sentence in extractive_highlights:
        index = cluster_sentences.index(sentence)
        indices_of_highlight_sentence.add(index)
    feature_Y_sum = np.sum(features[:, list(indices_of_highlight_sentence)], axis=1) #for likelihood and grad_likelihood
    feature_Y_sums.append(feature_Y_sum)


#Average Likelihood function
def f(theta, train_set):
    likelihoods = []
    for i in train_set:
        L = compute_conditional_kernel(theta, feature_matrices[i], similarity_matrices[i])
        ll = log_likelihood(theta, L, feature_Y_sums[i])
        likelihoods.append(ll)
    return -np.average(likelihoods)


#Average Grad_likelihood
def fprime(theta, train_set):
    gradients = []
    for i in train_set:
        L = compute_conditional_kernel(theta, feature_matrices[i], similarity_matrices[i])
        grad_ll = grad_log_likelihood(L, feature_matrices[i], feature_Y_sums[i])
        gradients.append(grad_ll)
    return -np.mean(gradients, axis=0)


#indices of articles in training set and test set
train_set = range(300)
test_set = range(300, 600, 1)

for _ in range(1):
    num_features = feature_matrices[0].shape[0]

    theta_0 = np.zeros(num_features) #initial theta for gradient descent
    print("optimizing")
    result = minimize(x0=theta_0, fun=f, jac=fprime, method="CG", args=(train_set,), options={"disp": True})

    theta = result.x

    print("This is the result of optimization: ", theta)

    # next we use the learned theta to produce highlights
    predictions = []
    random_predictions = []
    first_line_predictions = []
    print("predicting")
    for i in test_set:
        file_name, articles, cluster_sentences, _, extractive_highlights = datapoints[i]

        # dpp model
        assert len(cluster_sentences) == feature_matrices[i].shape[1]

        predicted_highlights = inference_summary_multi.predict(theta, feature_matrices[i], similarity_matrices[i],
                                                               cluster_sentences, len(extractive_highlights))
        result_point = ResultPoint(file_name, extractive_highlights, predicted_highlights)
        predictions.append(result_point)

        # randomly select sentences model
        random_prediction = ResultPoint(file_name, extractive_highlights,
                                        random.sample(cluster_sentences,
                                                      min(len(cluster_sentences), len(extractive_highlights))))
        random_predictions.append(random_prediction)

        # pick the first few sentences in order model
        first_line_predictions.append(ResultPoint(file_name, extractive_highlights, [a[0] for a in articles]))


        ## Useful code to compare articles, extracted highlights and predicted highlights
        # if i < 10:
        #     print(f"Cluster {i}")
        #     for j, article in enumerate(articles):
        #         print(f"Article {j}")
        #         for line_num, line in enumerate(article):
        #             print(line)
        #             if line_num >= 2:
        #                 print("...")
        #                 break
        #     print("Extractive Summaries")
        #     for summary in extractive_highlights:
        #         print(summary)
        #     print("Predicted Summaries")
        #     for summary in predicted_highlights:
        #         print(summary)
        #     print("")

    def report_performance(preds, name):
        scores = []
        for p in preds:
            extracted = [tuple(h) for h in p.extractive_highlights]
            predicted = [tuple(h) for h in p.predicted_highlights]
            score = len(set(extracted) & set(predicted)) / len(extracted)
            scores.append(score)

        performance = np.average(scores)
        print(f"{name} Model Performance is {performance:0.0%} accuracy")

    report_performance(predictions, "DPP")
    report_performance(random_predictions, "Random")
    report_performance(first_line_predictions, "First N")
    break