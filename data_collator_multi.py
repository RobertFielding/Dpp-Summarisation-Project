import json
import os
from data_cleaner_multi_news import highlight_to_extractive_summary
import nltk.data

from collections import namedtuple

print("loading sentence detector")
sent_detector = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
print("finished loading sentence detector")

ArticleCluster = namedtuple('ArticleCluster', 'file_name articles cluster_sentences highlights extractive_highlights')

directory = '.'
filename = 'train_na=4.json'


def load_datapoints_multi(num_files=None):
    with open(os.path.join(directory, filename)) as f:
        # this is the filename of the processed cluster - article cluster of length 4
        tokenized_output = []
        for line in f:
            cluster = json.loads(line)
            # cluster is dictionary {"ARTICLES": articles, "SUMMARY": summary}
            raw_articles = cluster["ARTICLES"]
            raw_summary = cluster["SUMMARY"]
            articles = []
            all_cluster_sentences = []
            for raw_article in raw_articles:
                article = [s.split() for s in sent_detector.tokenize(raw_article)][:30]  # keep at most 20 sentences per article
                articles.append(article)
                for sentence in article:
                    all_cluster_sentences.append(sentence)
            highlights = [s.split() for s in sent_detector.tokenize(raw_summary)][:5] #keep first 5 highlights
            extractive_highlights = highlight_to_extractive_summary(all_cluster_sentences, highlights)
            cluster_point = ArticleCluster(filename, articles, all_cluster_sentences, highlights, extractive_highlights)
            tokenized_output.append(cluster_point)
            if num_files is not None and len(tokenized_output) == num_files:
                # selects first num_files in the dataset if not all required
                break
    return tokenized_output


## Useful print statement

# if __name__ == "__main__":
#     datapoints = load_datapoints_multi(20)
#     for point in datapoints:
#         print(point.file_name)
#         for i, article in enumerate(point.articles):
#             print("Article", i, "Word_length", sum(len(s) for s in article), ':', article)
#
#         for i, (highlight, extracted_highlight) in enumerate(zip(point.highlights, point.extractive_highlights)):
#             print(f"Highlight {i}")
#             print(f"Actual Highlight   : {highlight}")
#             print(f"Extracted Highlight: {extracted_highlight}")

            #for poster
            # highlight_as_string = " ".join(highlight)
            # extracted_highlight_as_string = " ".join(extracted_highlight)
            # print(f"Actual Highlight   : {highlight_as_string}")
            # print(f"Extracted Highlight: {extracted_highlight_as_string}")

