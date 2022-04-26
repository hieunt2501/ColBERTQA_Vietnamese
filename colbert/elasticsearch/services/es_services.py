import hashlib
import time
import random


from colbert.elasticsearch.config.config import Config
from colbert.elasticsearch.services.base_service import BaseService
from elasticsearch import Elasticsearch

class ElasticSearchService(BaseService):
    def __init__(self, config: Config):
        super(ElasticSearchService, self).__init__(config)
        self.es = Elasticsearch(hosts=self.config.es_hosts)
        self.create_index()

    def create_index(self):
        request_body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "wiki_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "trim",
                                "lowercase",
                                "vi_stop_words_filter",
                            ]
                        }
                    },
                    "filter": {
                        "vi_stop_words_filter": {
                            "type": "stop",
                            "ignore_case": "true",
                            "stopwords": self.config.stop_words
                        }
                    }
                },
                "number_of_shards": 3,
                "number_of_replicas": 2
            }
        }
        done = False
        while not done:
            try:
                if not self.es.indices.exists(index=self.config.es_doc_index):
                    self.es.indices.create(
                        index=self.config.es_doc_index, body=request_body)
                    
                if not self.es.indices.exists(index=self.config.es_q_index):
                    self.es.indices.create(
                        index=self.config.es_q_index, body=request_body)
                done = True
                print("INDEX CREATED")
            except Exception as e:
                self.es = Elasticsearch(hosts=self.config.es_hosts)
                print(e)
                time.sleep(1)

    @staticmethod
    def generate_id(data):
        data_str = str(data).encode("utf8")
        result = hashlib.md5(data_str)
        return result.hexdigest()

    def insert(self, data, type="doc"):
        doc_id = self.generate_id(data)
        if type == "doc":
            index = self.config.es_doc_index
        else:
            index = self.config.es_q_index
            
        def save():
            self.es.index(index=index,
                          id=doc_id, document=data)

        def fail_call_back():
            self.es = Elasticsearch(hosts=self.config.es_hosts)

        self.make_request(save, fail_call_back)

    def delete(self, data, type="doc"):
        doc_id = self.generate_id(data)
        
        if type == "doc":
            index = self.config.es_doc_index
        else:
            index = self.config.es_q_index
            
        self.es.delete(index=index, id=doc_id)

    def search_question(self, question_id, size=1):
        def search_func():
            out = self.es.search(index=self.config.es_q_index,
                                 size=size,
                                 query={
                                     "bool": {
                                         "must": {"match": {"question_id": question_id}}
                                     }
                                 })
            return out["hits"]["hits"]
        results = self.make_request(search_func)
        outputs = []
        for item in results:
            outputs.append([item["_source"]["question_id"], item["_source"]["text"]])

        # outputs.sort(key=lambda i: i[0], reverse=True)
        return outputs

    def search_document(self, doc_id, size=1):
        def search_func():
            out = self.es.search(index=self.config.es_doc_index,
                                 size=size,
                                 query={
                                     "bool": {
                                          "must": {"match": {"doc_id": doc_id}}
                                     }
                                 })
            return out["hits"]["hits"]
        results = self.make_request(search_func)
        outputs = []
        for item in results:
            outputs.append([item["_source"]["doc_id"], 
                            item["_source"]["pos_passage"], 
                            item["_source"]["neg_passage"]])

        # outputs.sort(key=lambda i: i[0], reverse=True)
        return outputs

    def get_random_negative_document(self, qid, text, size=500):
        def search_func():
            out = self.es.search(index=self.config.es_doc_index,
                                 size=size,
                                 query={
                                     "bool":{
                                         "must_not": {"match": {"doc_id": qid}},
                                         "should": {"match": {"neg_passage": text}}
                                     }
                                 })
            return out["hits"]["hits"]
        results = self.make_request(search_func)
        outputs = []
        for item in results:
            outputs.append([item["_score"], item["_source"]["neg_passage"]])
        outputs.sort(key=lambda i: i[0], reverse=True)
        chosen_passage = random.choice(outputs)
        return chosen_passage[1]

    def get_random_questions(self, size=1):
        def search_func():
            out = self.es.search(index=self.config.es_q_index,
                                 size=size,
                                 query={
                                     "function_score": {
                                         "functions":[{
                                             "random_score": {}}
                                         ]
                                    }
                                 })
            
            return out["hits"]["hits"]
        
        results = self.make_request(search_func)
        outputs = []
        for item in results:
            outputs.append([item["_source"]["question_id"], item["_source"]["text"]])
        return outputs