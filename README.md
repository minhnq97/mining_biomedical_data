# Data mining on Vietnamese Medical data
## 1. Setup elasticsearch:

Install elasticsearch by running ```./setup.sh```

Then run ```docker-compose up -d```

Check if the elastic work or not by running ```curl -X GET "localhost:9200/_cat/nodes?v&pretty"```

## 2. Flask API: 
Start the server by running ```python start.py```

API for extract **word cloud**:

- request: 

```curl -XGET "localhost:5000/word_cloud?category_id=c5&top_n=7"```

- response:

```[{"key": "viêm", "doc_count": 61}, {"key": "sơ", "doc_count": 51}, {"key": "gan", "doc_count": 49}, {"key": "hp", "doc_count": 44}, {"key": "cháu", "doc_count": 40}, {"key": "thuốc", "doc_count": 35}, {"key": "architect", "doc_count": 25}]```

API for extract **score similarity**:

- request (doing): 

```send a sentence/document```

- response (doing):

```[
      {
        "_index" : "new_medlatec",
        "_type" : "_doc",
        "_id" : "o6r5uXQBUJQDowV2xly0",
        "_score" : 1.0,
        "_source" : {
          "question" : """Chào bác sĩ, 
Em vừa làm xét nghiệm HbsAg là 1068 COI và HBsAb định lượng <0,2 U/L  kết quả cho dương tính thì không biết em có phải thực hiện điều trị hay uống thuốc gì không ạ? 
Em cảm ơn ạ!""",
          "preprocessed_question" : "chào bác sĩ em vừa làm xét nghiệm hbsag là  coi và hbsab định lượng   u l kết quả cho dương tính thì không biết em có phải thực hiện điều trị hay uống thuốc gì không ạ em cảm ơn ạ",
          "category_id" : "c5",
          "category" : "tiêu hóa - gan mật",
          "keywords" : [
            "coi hbsab",
            "hbsag coi",
            "hbsab",
            "coi",
            "hbsag"
          ]
        }
      },
      {
        "_index" : "new_medlatec",
        "_type" : "_doc",
        "_id" : "pKr5uXQBUJQDowV2xly0",
        "_score" : 1.0,
        "_source" : {
          "question" : """Bác sĩ cho em hỏi 2 thuốc này em thay đổi cho nhau được ko ạ. 
Em cảm ơn bác sĩ""",
          "preprocessed_question" : "bác sĩ cho em hỏi  thuốc này em thay đổi cho nhau được không ạ em cảm ơn bác sĩ",
          "category_id" : "c5",
          "category" : "tiêu hóa - gan mật",
          "keywords" : [
            "thuốc"
          ]
        }
      }
    ]
