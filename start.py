#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 minhnq <minhnq@rd04>
#
# Distributed under terms of the MIT license.

"""

"""
from flask import Flask
app = Flask(__name__)

@app.route('/create')
def insert_question():
    # classify category
    return "hello"

@app.route('/similarity')
def score_similarity():
    return "hello"

if __name__ == "__main__":
    app.run()
