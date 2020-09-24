#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 minhnq <minhnq@rd04>
#
# Distributed under terms of the MIT license.

from extract_keyword import preload_model, get_keyword
from preprocess import _format_line, load_data, merge_data_by_category, drop_category, redefine_data
import json
from elasticsearch.helpers import bulk
from elasticsearch  import Elasticsearch

def gen_bulk(all_data, index_name="index_name"):
    cate_dict={}
    stopwords, tfidf_vectorizer, feature_names = preload_model()
    with open("mapping_category.txt", "rt") as f:
        for line in f.read().splitlines():
            id, cate = line.split(None,1)
            cate_dict[id] = cate
    for data, label in all_data:
        preprocessed_q = _format_line(data)
        top_5_keywords = list(get_keyword(preprocessed_q,stopwords,tfidf_vectorizer,feature_names).keys())[:5]
        yield {
            "_index": index_name,
            "question": data,
            "preprocessed_question":preprocessed_q,
            "category_id": label,
            "category":cate_dict[label],
            "keywords": top_5_keywords,
        }

def create_bulk():
    all_data = load_data("crawl_md_question/")
    all_data, label_list = merge_data_by_category(all_data, target_merge={'c22':'c22', 'c57':'c22', 'c55':'c22',
                                                                         'c11':'c22','c33':'c22','c10':'c22','c34':'c22', # c22: nội khoa
                                                                         'c5':'c5','c3':'c5','c21':'c21','c19':'c21','c4':'c21'}) #c5: viêm gan; c21: sản phụ khoa
    all_data, label_list = drop_category(all_data, drop_target=['c20','c17','c73'])
    all_data, label_list = redefine_data(all_data, threshold=50)
    return all_data, label_list

def extract_json_bulk():
    all_data, label_list = create_bulk()
    cate_dict = {}
    result=[]
    stopwords, tfidf_vectorizer, feature_names = preload_model()
    with open("mapping_category.txt", "rt") as f:
        for line in f.read().splitlines():
            id, cate = line.split(None, 1)
            cate_dict[id] = cate
    index=0
    for data, label in all_data:
        index+=1
        preprocessed_q = _format_line(data)
        top_5_keywords = list(get_keyword(preprocessed_q, stopwords, tfidf_vectorizer, feature_names).keys())[:5]
        result.append(json.dumps({"index": {"_id": str(index)}}))
        result.append(json.dumps({
            "question": data.replace("\n", " "),
            "preprocessed_question": preprocessed_q,
            "category_id": label,
            "category": cate_dict[label],
            "keywords": top_5_keywords,
        }, ensure_ascii=False))

    with open("es_bulk_data.json", "w+") as f:
        f.writelines("\n".join(result))
    pass

if __name__ == '__main__':
    extract_json_bulk()
    # elastic_client = Elasticsearch(hosts=["localhost"])
    # all_data, label_list = create_bulk()
    # bulk(elastic_client,gen_bulk(all_data, index_name="new_medlatec"))
