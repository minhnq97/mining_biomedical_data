#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 minhnq <minhnq@rd04>
#
# Distributed under terms of the MIT license.

"""

"""
from flask import Flask, request, g
from elasticsearch import Elasticsearch
import json
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options
import torch
import numpy as np
from fairseq.models.roberta import RobertaModel
from preprocess import _format_line
import pickle
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/classifier', methods=["POST"])
def text_classifier_category():
    data = json.loads(request.data)
    question = data["question"]
    tfidf_q = g.vectorizer.transform([question])
    category_id = g.svm_model.predict(tfidf_q)
    return json.dumps({"category_id":category_id[0], "category_name":g.mapping_cid[category_id[0]]}, ensure_ascii=False)

@app.route("/similarity", methods=["POST"])
def score_similarity():
    data = json.loads(request.data)
    question = data["question"]
    top_n = int(data["top_n"])
    preprocessed_q = _format_line(question)
    subwords = g.phobert.encode(preprocessed_q)
    if len(subwords) > 256:
        subwords = subwords[:255]
    last_layer_features = g.phobert.extract_features(subwords)
    full_vectors = last_layer_features.cpu().detach().numpy()
    embedding_vector = full_vectors[0, 0, :]

    result = g.es_client.search(index="new_medlatec_dummy", body={
      "size":top_n,
      "query": {
          "script_score": {
            "query": {"match_all": {}},
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
              "params":  {"query_vector": embedding_vector}
            }
          }
      }
    })

    hits_result = result['hits']['hits']
    final_result = []
    for res in hits_result:
        del res['_source']['question_vector'] # No return dense vector for visual efficiency
        final_result.append(res)
    return json.dumps(final_result, ensure_ascii=False)

@app.route("/word_cloud", methods=["GET"])
def search_word_cloud():
    category_id = request.args.get('category_id')
    top_n_word = int(request.args.get('top_n',10))
    result = g.es_client.search(index="new_medlatec_dummy", body={
        "aggs": {
            "result": {
                "terms": {
                    "field": "keywords.keyword",
                    "order": {
                        "_count": "desc"
                    },
                    "size": top_n_word
                }
            }
        },
        "size": 0,
        "query": {
            "bool": {
                "must": [],
                "filter": [
                    {
                        "term": {"category_id": category_id}
                    }
                ]
            }
        }
    })
    final_res = result["aggregations"]["result"]["buckets"]
    return json.dumps(final_res, ensure_ascii=False)

@app.before_request
def initialize_model():
    g.es_client = Elasticsearch(hosts=["localhost"])
    g.phobert = RobertaModel.from_pretrained('model/',
                                           checkpoint_file='model.pt')
    g.phobert.eval()  # disable dropout (or leave in train mode to finetune)
    g.phobert = g.phobert.to(torch.device("cuda:1")) # fix code cuda device
    parser = options.get_preprocessing_parser()
    parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
                        default="model/bpe.codes")
    args = parser.parse_args()
    g.phobert.bpe = fastBPE(args)  # Incorporate the BPE encoder into PhoBERT
    g.vectorizer = pickle.load(open("model/tfidf_model.pkl", 'rb'))
    g.svm_model = pickle.load(open("model/svm_model.pkl", 'rb'))
    g.mapping_cid={}
    with open("resources/mapping_category.txt", "rt") as f:
        for line in f.read().splitlines():
            id, name = line.split(None, 1)
            g.mapping_cid[id] = name



if __name__ == "__main__":
    app.run()
