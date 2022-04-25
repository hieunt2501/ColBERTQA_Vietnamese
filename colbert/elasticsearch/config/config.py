import os

from colbert.elasticsearch.common.common_key import *


class Config:
    es_hosts = [item.strip() for item in os.getenv(ELASTIC_HOSTS, "http://172.28.0.23:20231,http://172.28.0.23:20232"
                                                                  ",http://172.28.0.23:20233").split(",")]
    es_q_index = os.getenv(ELASTIC_DOC_IDX, "colbert_question")
    es_doc_index = os.getenv(ELASTIC_DOC_IDX, "colbert_document")
    doc_qa_url = os.getenv(DOC_QA_URL, "http://172.28.0.23:35330/qa")
    vn_core_url = os.getenv(VN_CORE_URL, "http://172.28.0.23")
    vn_core_port = int(os.getenv(VN_CORE_PORT, 20215))
    text_max_length = int(os.getenv(TEXT_MAX_LENGTH, 400))
    stop_words = [item.rstrip() for item in open("elasticsearch/config/question_word.txt", "r", encoding="utf8").readlines()]
