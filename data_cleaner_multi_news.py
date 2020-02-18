import os
import json
import numpy as np

directory = r"C:\Users\Rober\Documents\Text Summarization\multi_news"
article_separator = "story_separator_special_tag"


def highlight_to_extractive_summary(article_cluster_sentences, highlights):
    X_bag_of_words = [set(sentence) for sentence in article_cluster_sentences]
    #produces list of sets of sentence words [{words in sentence}article cluster]
    highlights_bag_of_words = [set(sentence) for sentence in highlights]
    #produces list of sets of highlight words [{words in highlight}article cluster]
    num_highlights = len(highlights)
    num_X_sentences = len(article_cluster_sentences)
    similarity_matrix = np.zeros((num_highlights, num_X_sentences))
    for i, highlight_words in enumerate(highlights_bag_of_words):
        for j, line_words in enumerate(X_bag_of_words):
            similarity_matrix[i, j] = len(line_words & highlight_words) / (0.5 * len(highlight_words) + 0.5 * len(line_words))
            #simple measure to convert abstractive highlight to extractive

    extractive_summaries = []
    for i, highlight in enumerate(highlights):
        sentence_similarity = similarity_matrix[i, :]
        idx = np.random.choice(np.flatnonzero(sentence_similarity == np.max(sentence_similarity)))
        #chooses a random sentence in article cluster with highest similarity to highlight (if multiple)
        extractive_highlight = article_cluster_sentences[idx]
        extractive_summaries.append(extractive_highlight)

    return extractive_summaries


def split(articles_file, summaries_file, name):
    #fct pairs articles and respective highlights in dictionary, saves dictionary in JSON based on number articles in cluster
    f_handles = {}
    #dictionary with key based on number of articles related to a news story
    with open(articles_file, encoding="utf-8") as f_articles, open(summaries_file, encoding="utf-8") as f_summaries:
        i = 0
        for article_cluster, summary in zip(f_articles, f_summaries):
            summary = summary.strip()
            articles = article_cluster.strip().split(article_separator)
            articles = [article for article in articles if len(article) > 3] #removes empty articles
            num_articles = len(articles)
            if num_articles not in f_handles:
                f_handles[num_articles] = open(f"./{name}_na={num_articles}.json", 'w', encoding="utf-8")
                #creates file if the number of articles in cluster not yet seen
            out_f = f_handles[num_articles]
            output = {"ARTICLES": articles, "SUMMARY": summary}
            json.dump(output, out_f, separators = (',', ':'))
            #stores to JSON file in directory saved to
            out_f.write("\n")
            #separates article cluster by new line

            i += 1
            if i % 10000 == 0:
                print(f"processed: {i} clusters")


if __name__ == '__main__':  # only run this code if this file is run as the main method
    for name in ["train", "test"]:
        split(os.path.join(directory, "train.txt.src.truncated"), os.path.join(directory, "train.txt.tgt.truncated"), name)

