# Data mining on Vietnamese Medical data
## 1. Setup elasticsearch:

Install elasticsearch by running ```./setup.sh```

Then run ```docker-compose up -d```

Check if the elastic work or not by running ```curl -X GET "localhost:9200/_cat/nodes?v&pretty"```

Insert bulk data: ```curl -s -H "Content-Type: application/json" -XPOST localhost:9200/{index_name}/docs/_bulk --data-binary "@es_bulk_data.json"```

## 2. Flask API: 
Start the server by running ```python start.py```

API for **classifying input** to medical categories:

- request:

```curl -X POST -H "Content-type:application/json"  -d '{"question": "viêm gan b"}' localhost:5000/classifier```

- response:

```
{
   "category_id":"c5",
   "category_name":"tiêu hóa - gan mật"
}
```

API for extract **word cloud**:

- request: 

```curl -XGET "localhost:5000/word_cloud?category_id=c5&top_n=7"```

- response:

```
[
   {
      "key":"viêm",
      "doc_count":61
   },
   {
      "key":"sơ",
      "doc_count":51
   },
   {
      "key":"gan",
      "doc_count":49
   },
   {
      "key":"hp",
      "doc_count":44
   },
   {
      "key":"cháu",
      "doc_count":40
   },
   {
      "key":"thuốc",
      "doc_count":35
   },
   {
      "key":"architect",
      "doc_count":25
   }
]
```

API for extract **score similarity**:

- request (doing): 

```curl -X POST -H "Content-type:application/json"  -d `{"question": "viêm gan b", "top_n":2
}` localhost:5000/similarity```


- response (doing):

```[
   {
      "_index":"new_medlatec_dummy",
      "_type":"_doc",
      "_id":"v6x1w3QBUJQDowV2MvbX",
      "_score":1.7523599,
      "_source":{
         "question":"Về viêm gan b mem gan và số lượng virus",
         "preprocessed_question":"về viêm gan b mem gan và số lượng virus",
         "category_id":"c22",
         "category":"dị ứng",
         "keywords":[
            "gan mem",
            "mem gan",
            "mem",
            "gan",
            "virus"
         ]
      }
   },
   {
      "_index":"new_medlatec_dummy",
      "_type":"_doc",
      "_id":"qq11w3QBUJQDowV2XQNm",
      "_score":1.7464817,
      "_source":{
         "question":"bệnh bướu cổ basedow",
         "preprocessed_question":"bệnh bướu cổ basedow",
         "category_id":"c21",
         "category":"nội tiết sinh dục",
         "keywords":[
            "basedow",
            "bệnh"
         ]
      }
   }
]
```
