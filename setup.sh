docker pull docker.elastic.co/elasticsearch/elasticsearch:7.5.2
# should run docker-compose instead
# docker run -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.5.2