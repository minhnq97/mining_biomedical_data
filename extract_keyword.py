""" Created by minhnq """
import _pickle as cPickle
import json
import os
from glob import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from CocCocTokenizer import PyTokenizer
# from create_es_bulk import create_bulk
from preprocess import _format_line
import pickle


def preload_model():
    data = pickle.load(open("raw.pkl", "rb"))
    stopwords = open("resources/src_resources_vietstopwords.txt").read().splitlines()

    # with open('model/tfidf_model.pkl', 'rb') as fin:
    #     tfidf_vectorizer = cPickle.load(fin)
    #     # inv_map = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()}
    #     feature_names = tfidf_vectorizer.get_feature_names()
    # return stopwords, tfidf_vectorizer, feature_names

    tfidf_vectorizer = train_tfidf_by_data(data, stopwords)
    feature_names = tfidf_vectorizer.get_feature_names()

    return  stopwords, tfidf_vectorizer, feature_names

def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


def get_data(list_file_data):
    data = []
    for file_path in list_file_data:
        with open(file_path, 'r', encoding='utf-8') as file_data:
            head = [next(file_data) for _ in range(500)]
        data.extend(head)
    return data


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def transform_ngram(text, phraser_list):
    line_words = text.split()

    for phraser in phraser_list:
        line_words = phraser[line_words]
    line_transform = " ".join(line_words)
    return line_transform


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keyword(raw_input,stopwords, tfidf_vectorizer):
    feature_names = tfidf_vectorizer.get_feature_names()
    T = PyTokenizer(load_nontone_data=True)
    clean_text = _format_line(raw_input)
    # clean_text = " ".join(T.word_tokenize(clean_text, tokenize_option=0))
    clean_text = " ".join([w for w in clean_text.split() if w not in stopwords])
    clean_text_tf_idf = tfidf_vectorizer.transform([clean_text])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(clean_text_tf_idf.tocoo())

    # extract only the top n; n here is 10
    keys = extract_topn_from_vector(feature_names, sorted_items, 20)
    return keys

def train_tfidf_by_data(data,stopwords):
    T = PyTokenizer(load_nontone_data=True)
    tokenizer = T.word_tokenize

    tfidf = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 1), max_df=0.7, min_df=20, stop_words=stopwords)
    tfidf.fit(data)
    return tfidf


if __name__ == "__main__":
    # ============ Only need for train tfidf =============
    # datas = glob("crawl_md_question/*.pt.json")
    # all_content = []
    # for i in datas:
    #     dt = json.load(open(i, "r"))
    #     content = [i['q_content'] for i in dt]
    #     all_content += content
    # print(len(all_content))
    # pickle.dump(all_content, open("raw.pkl", "wb"))
    # data = pickle.load(open("raw.pkl", "rb"))
    # stopwords = open("resources/src_resources_vietstopwords.txt").read().splitlines()
    #
    # tfidf = train_tfidf_by_data(data, stopwords)
    # ====================================================
    stopwords, tfidf_vectorizer, feature_names = preload_model()
    keywords = get_keyword("Chi phí xét nghiệm phát hiện bệnh ung thư cổ tử cung là bao nhiêu?",stopwords, tfidf_vectorizer)
    print(keywords)
