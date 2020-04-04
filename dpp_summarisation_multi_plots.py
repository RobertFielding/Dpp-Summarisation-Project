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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


ResultPoint = namedtuple('ResultPoint', 'file_name extractive_highlights predicted_highlights')
num_datapoints = 260
upper_bound_train_cases = 60
print("started loading data")
datapoints = load_datapoints_multi(num_datapoints)
print("finished loading data")
feature_matrices = []
similarity_matrices = []
feature_Y_sums = []


inv_doc_freq_file = r'train_na=4_inv_doc_freq.json'

#either calculates inverse_document_frequency or opens stored file
print("start inv_doc")
if not os.path.isfile(inv_doc_freq_file):
    inverse_document_frequency_dict = inverse_document_frequency.compute_and_save(datapoints, inv_doc_freq_file)
else:
    with open(inv_doc_freq_file) as f:
        inverse_document_frequency_dict = json.load(f)
print("finish inv_doc")


similarity_matrices = compute_and_save_similarity(datapoints, inverse_document_frequency_dict)
feature_matrices = compute_and_save_features(datapoints, similarity_matrices)


for i, datapoint in enumerate(datapoints[: upper_bound_train_cases]):
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
test_set = range(upper_bound_train_cases, num_datapoints, 1)
num_features = feature_matrices[0].shape[0]
performance_list_DPP = []
performance_list_random = []
performance_list_first_sentence = []
performance_list_DPP_train = []
performance_list_random_train = []
performance_list_first_sentence_train = []


for num_training_points in range(1, upper_bound_train_cases + 1, 1):
    train_set = range(num_training_points)

    theta_0 = np.zeros(num_features) #initial theta for gradient descent
    print("optimizing")
    result = minimize(x0=theta_0, fun=f, jac=fprime, method="CG", args=(train_set,), options={"disp": True})

    theta = result.x
    print("This is the result of optimization: ", theta)

    # next we use the learned theta to produce highlights
    predictions = []
    random_predictions = []
    first_line_predictions = []
    predictions_train = []
    random_predictions_train = []
    first_line_predictions_train = []
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

    def report_performance(preds):
        scores = []
        for p in preds:
            extracted = [tuple(h) for h in p.extractive_highlights]
            predicted = [tuple(h) for h in p.predicted_highlights]
            score = len(set(extracted) & set(predicted)) / len(extracted)
            scores.append(score)
        performance = np.average(scores)
        return performance

    performance_DPP = report_performance(predictions)
    performance_list_DPP.append(performance_DPP)
    performance_random = report_performance(random_predictions)
    performance_list_random.append(performance_random)
    performance_first_sentence = report_performance(first_line_predictions)
    performance_list_first_sentence.append(performance_first_sentence)



    for j in train_set:
        file_name, articles, cluster_sentences, _, extractive_highlights = datapoints[j]

        # dpp model
        assert len(cluster_sentences) == feature_matrices[j].shape[1]

        predicted_highlights_train = inference_summary_multi.predict(theta, feature_matrices[j], similarity_matrices[j],
                                                               cluster_sentences, len(extractive_highlights))

        result_point_train = ResultPoint(file_name, extractive_highlights, predicted_highlights_train)
        predictions_train.append(result_point_train)

        # randomly select sentences model
        random_prediction_train = ResultPoint(file_name, extractive_highlights,
                                        random.sample(cluster_sentences,
                                                      min(len(cluster_sentences), len(extractive_highlights))))
        random_predictions_train.append(random_prediction_train)

        # pick the first few sentences in order model
        first_line_predictions_train.append(ResultPoint(file_name, extractive_highlights, [a[0] for a in articles]))

    def report_performance(preds):
        scores = []
        for p in preds:
            extracted = [tuple(h) for h in p.extractive_highlights]
            predicted = [tuple(h) for h in p.predicted_highlights]
            score = len(set(extracted) & set(predicted)) / len(extracted)
            scores.append(score)
        performance = np.average(scores)
        return performance

    performance_DPP_train = report_performance(predictions_train)
    performance_list_DPP_train.append(performance_DPP_train)
    performance_random_train = report_performance(random_predictions_train)
    performance_list_random_train.append(performance_random_train)
    performance_first_sentence_train = report_performance(first_line_predictions_train)
    performance_list_first_sentence_train.append(performance_first_sentence_train)

x_points_to_plot = range(upper_bound_train_cases)
y1_points_to_plot = performance_list_DPP
plt.plot(x_points_to_plot, y1_points_to_plot, label="DPP", c='blue')
y2_points_to_plot = performance_list_random
plt.plot(x_points_to_plot, y2_points_to_plot, label="Random", c='orange')
y3_points_to_plot = performance_list_first_sentence
plt.plot(x_points_to_plot, y3_points_to_plot, label="First Sentence", c='green')
plt.legend()
plt.xlabel("Number of Training Clusters")
plt.ylabel("% of Right Sentences Picked")
ax = plt.axes()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())



x_points_to_plot = range(upper_bound_train_cases)
y1_points_to_plot = performance_list_DPP_train
plt.plot(x_points_to_plot, y1_points_to_plot, label="DPP", linestyle    ='dotted', c='blue')
y2_points_to_plot = performance_list_random_train
plt.plot(x_points_to_plot, y2_points_to_plot, label="Random", linestyle='dotted', c='orange')
y3_points_to_plot = performance_list_first_sentence_train
plt.plot(x_points_to_plot, y3_points_to_plot, label="First Sentence", linestyle='dotted', c='green')
plt.show()