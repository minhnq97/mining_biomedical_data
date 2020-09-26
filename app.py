from flask import Flask
import json

app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route("/classifier", methods=["POST"])
def text_classifier_category():
    return json.dumps({
        "category_id": "c5",
        "category_name": "tiêu hóa - gan mật"
    })


@app.route("/similarity", methods=["POST"])
def score_similarity():
    return json.dumps([{
        "_index": "new_medlatec_dummy",
        "_type": "_doc",
        "_id": "v6x1w3QBUJQDowV2MvbX",
        "_score": 1.7523599,
        "_source": {
            "question": "Về viêm gan b mem gan và số lượng virus",
            "preprocessed_question": "về viêm gan b mem gan và số lượng virus",
            "category_id": "c22",
            "category": "dị ứng",
            "keywords": [
                "gan mem",
                "mem gan",
                "mem",
                "gan",
                "virus"
            ]
        }
    },
        {
            "_index": "new_medlatec_dummy",
            "_type": "_doc",
            "_id": "qq11w3QBUJQDowV2XQNm",
            "_score": 1.7464817,
            "_source": {
                "question": "bệnh bướu cổ basedow",
                "preprocessed_question": "bệnh bướu cổ basedow",
                "category_id": "c21",
                "category": "nội tiết sinh dục",
                "keywords": [
                    "basedow",
                    "bệnh"
                ]
            }
        }
    ])


@app.route("/word_cloud", methods=["GET"])
def search_word_cloud():
    return json.dumps(
        [{"key": "viêm", "doc_count": 61}, {"key": "sơ", "doc_count": 51}, {"key": "gan", "doc_count": 49},
         {"key": "hp", "doc_count": 44}, {"key": "cháu", "doc_count": 40}, {"key": "thuốc", "doc_count": 35},
         {"key": "architect", "doc_count": 25}])


if __name__ == '__main__':
    app.run()
