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

app = Flask(__name__)

@app.route('/create')
def insert_question():
    # classify category
    return "hello"

@app.route('/similarity')
def score_similarity():
    return "hello"

@app.route("/word_cloud", methods=["GET"])
def search_word_cloud():
    category_id = request.args.get('category_id')
    top_n_word = int(request.args.get('top_n',10))
    result = g.es_client.search(index="new_medlatec", body={
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
    return json.dumps(final_res)

@app.before_request
def initialize_model():
    g.es_client = Elasticsearch(hosts=["localhost"])


if __name__ == "__main__":
    app.run()
