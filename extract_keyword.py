""" Created by minhnq """
import _pickle as cPickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize

from preprocess import _format_line


def preload_model():
    stopwords = get_stop_words("resources/src_resources_vietstopwords.txt")
    # open("resources/src_resources_vietstopwords.txt", 'r', encoding='utf-8').read().split("\n")

    with open('model/tfidf_medlatec.pkl', 'rb') as fin:
        tfidf_vectorizer = cPickle.load(fin)
        # inv_map = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()}
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


def get_keyword(raw_input,stopwords, tfidf_vectorizer,feature_names):

    clean_text = _format_line(raw_input)
    clean_text = word_tokenize(clean_text, format="text")
    clean_text = " ".join([w for w in clean_text.split() if w not in stopwords])
    clean_text_tf_idf = tfidf_vectorizer.transform([clean_text])

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(clean_text_tf_idf.tocoo())

    # extract only the top n; n here is 10
    keys = extract_topn_from_vector(feature_names, sorted_items, 20)
    return keys

def train_tfidf(train_texts):
    if os.path.exists("model/tfidf_medlatec.pkl") is False:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1, 2))
        tfidf = vectorizer.fit(train_texts)
        import pickle
        pickle.dump(tfidf, open("model/tfidf_medlatec.pkl", "wb"))
    pass


if __name__ == "__main__":
    # ============ Only need for train tfidf =============
    # all_data, _ = create_bulk()
    # full_text = []
    # for data,label in all_data:
    #     full_text.append(_format_line(data))
    #
    # train_tfidf(full_text)
    # ====================================================
    stopwords, tfidf_vectorizer, feature_names = preload_model()
    keywords = get_keyword("'Chào bác sĩ, Em vừa làm xét nghiệm HbsAg là 1068 COI và HBsAb định lượng <0,2 U/L  "
                           "kết quả cho dương tính thì không biết em có phải thực hiện điều trị hay uống thuốc gì không ạ? Em cảm ơn ạ!'",stopwords, tfidf_vectorizer,feature_names)
    print(keywords)
