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
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options
import torch
import numpy as np
from fairseq.models.roberta import RobertaModel

def gen_bulk(client, all_data, index_name="index_name", model=None):
    cate_dict = {}
    stopwords, tfidf_vectorizer, feature_names = preload_model()
    with open("mapping_category.txt", "rt") as f:
        for line in f.read().splitlines():
            id, cate = line.split(None, 1)
            cate_dict[id] = cate
    index = 0
    for data, label in all_data:
        index += 1
        preprocessed_q = _format_line(data)
        top_5_keywords = list(get_keyword(preprocessed_q, stopwords, tfidf_vectorizer, feature_names).keys())[:5]
        subwords = model.encode(preprocessed_q)
        if len(subwords) > 256:
            subwords = subwords[:255]
        last_layer_features = model.extract_features(subwords)
        full_vectors = last_layer_features.cpu().detach().numpy()
        embedding = full_vectors[0, 0, :]
        request = {
            "_op_type": "index",
            "_index": index_name,
            "question": data,
            "preprocessed_question": preprocessed_q,
            "category_id": label,
            "category": cate_dict[label],
            "keywords": top_5_keywords,
            "question_vector": embedding.tolist()
        }
        bulk(client, request)

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

def extract_json_bulk_with_embedding(model):
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
        subwords = model.encode(preprocessed_q)
        if len(subwords) > 256:
            subwords = subwords[:255]
        last_layer_features = model.extract_features(subwords)
        full_vectors = last_layer_features.cpu().detach().numpy()
        embedding = full_vectors[0,0,:]
        result.append(json.dumps({"index": {"_id": str(index)}}))
        result.append(json.dumps({
            "question": data.replace("\n", " "),
            "preprocessed_question": preprocessed_q,
            "category_id": label,
            "category": cate_dict[label],
            "keywords": top_5_keywords,
            "question_vector": embedding.tolist()
        }, ensure_ascii=False))

    with open("es_bulk_data.json", "w+") as f:
        f.writelines("\n".join(result))
    pass

def gen_data_from_json(index_name="medlatec_dummy"):
    with open("es_bulk_data.json", "rt") as f:
        for index, line in enumerate(f.read().splitlines()):
            request = json.loads(line)
            if "index" in request.keys():
                continue
            request["_index"] = index_name
            yield request


if __name__ == '__main__':

    # phobert = RobertaModel.from_pretrained('/mnt/VAIS/Home/minhnq/nlp_tools/PhoBERT_base_fairseq',
    #                                        checkpoint_file='model.pt')
    # phobert.eval()  # disable dropout (or leave in train mode to finetune)
    # phobert = phobert.to(torch.device("cuda:1"))
    # parser = options.get_preprocessing_parser()
    # parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
    #                     default="/mnt/VAIS/Home/minhnq/nlp_tools/PhoBERT_base_fairseq/bpe.codes")
    # args = parser.parse_args()
    # phobert.bpe = fastBPE(args)  # Incorporate the BPE encoder into PhoBERT
    # extract_json_bulk_with_embedding(phobert)
    # all_data, label_list = create_bulk()
    # gen_bulk(elastic_client, all_data, index_name="new_medlatec", model=phobert)
    INDEX_NAME = "new_medlatec_dummy"
    with open("index.json") as index_file:
        source = index_file.read().strip()
    elastic_client = Elasticsearch(hosts=["localhost"], timeout=30)
    elastic_client.indices.create(index=INDEX_NAME, body=source)
    bulk(elastic_client, gen_data_from_json(index_name=INDEX_NAME))
