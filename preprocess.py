#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 minhnq <minhnq@rd04>
#
# Distributed under terms of the MIT license.

"""

"""
import re
import json
import seaborn as sns
import matplotlib.pyplot as plt
from underthesea import word_tokenize
from underthesea import pos_tag
from tqdm import tqdm

list_index_data = [3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 39, 55,
                       56, 57, 58, 60, 61, 62, 64, 73, 74, 76]

REPLACE_SHORT_WORD = {"e":"em", "k": "không", "ko": "không", "xn": "xét nghiệm", "bv":"bệnh viện", "bsi": "bác sĩ", "bs": "bác sĩ"}
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@.,;\-\'\\n]')
BAD_SYMBOLS_RE = re.compile('[~!@#$%^&*-+=?:]')


def _format_line(sentence):
    sentence = sentence.lower()
    sentence = REPLACE_BY_SPACE_RE.sub(' ', sentence)
    sentence = BAD_SYMBOLS_RE.sub('', sentence)
    sentence = ' '.join([REPLACE_SHORT_WORD[word] if word in REPLACE_SHORT_WORD else word for word in sentence.split()])
    sentence = ' '.join(["0" if word.isnumeric() else word for word in sentence.split()])
    sentence = ' '.join(["" if any(map(str.isdigit, word)) is True else word for word in sentence.split()])
    # sentence = word_tokenize(sentence, format="text")
    # tags = pos_tag(sentence)
    # sentence = ' '.join([ w  for w,t in tags if t not in['C', 'Cc', 'CH', 'E', 'I', 'L', 'P', 'R', 'T' ]])

    return sentence

def load_data(root_dir=""):
    all_data=[]
    for index in list_index_data:
        with open(root_dir + "c"+ str(index) + ".pt.json", "rt") as f:
            batch_data = json.load(f)
            for data in tqdm(batch_data):
              all_data.append((data['q_content'], data['c_id']))
    return all_data

def redefine_data(all_data, threshold=20):
    new_data=[]
    label_list=[]
    label_freq = {}
    for data in all_data:
        c_id = data[1]
        if c_id not in label_freq:
            label_freq[c_id] = 1
        else:
            label_freq[c_id] += 1

    for data in all_data:
        c_id = data[1]
        if label_freq[c_id] > threshold:
            new_data.append(data)
            label_list.append(c_id)

    return new_data, list(set(label_list))

def merge_data_by_category(all_data, target_merge=None):
    if target_merge is None:
        return all_data, None
    else:
        new_data=[]
        label_list = []
        for data in all_data:
            c_id = data[1]
            if c_id in target_merge:
                data = (data[0],target_merge[c_id])
            new_data.append(data)
            label_list.append(c_id)
        return new_data, list(set(label_list))

def drop_category(all_data, drop_target=None):
    if drop_target is None:
        return all_data, None
    else:
        new_data=[]
        label_list = []
        for data in all_data:
            c_id = data[1]
            if c_id not in drop_target:
                new_data.append(data)
                label_list.append(c_id)
        return new_data, list(set(label_list))

#all_data = load_data()
#all_data, label_list = merge_data_by_category(all_data, target_merge={'c22':'c22', 'c57':'c22', 'c55':'c22',
#                                                                      'c11':'c22','c33':'c22','c10':'c22','c34':'c22', # c22: nội khoa
#                                                                      'c5':'c5','c3':'c5','c21':'c21','c19':'c21','c4':'c21'}) #c5: viêm gan; c21: sản phụ khoa
#all_data, label_list = drop_category(all_data, drop_target=['c20','c17','c73'])
#all_data, label_list = redefine_data(all_data, threshold=50)


