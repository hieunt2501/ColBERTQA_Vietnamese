import os
import ujson
import random
from tqdm import tqdm

from underthesea import sent_tokenize, word_tokenize

# import sys
# sys.path.insert(1, "../")
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument('--query_maxlen', dest='query_maxlen', default=500, type=int)
# parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=1024, type=int)
# parser.add_argument('--pretrained_tokenizer', dest='pretrained_tokenizer', default="../pretrained/pretrained/bartpho")
# parser.add_argument('--bsize', dest='bsize', default=10, type=int)

# parser.add_argument('--accum', dest='accumsteps', default=2, type=int)
# parser.add_argument('--fine_tune', dest='fine_tune', default=False, action='store_true')

# parser.add_argument('--queries', dest='queries', default="../dataset/pretrain/pseudo_questions.tsv")
# parser.add_argument('--positive_collection', dest='positive_collection', default="../dataset/pretrain/positive_collection.tsv")
# parser.add_argument('--negative_collection', dest='negative_collection', default="../dataset/pretrain/negative_collection.tsv")
# parser.add_argument('--top500', dest='top500', default="../dataset/pretrain/top500_per_line.txt")

# args = parser.parse_args()

from ast import literal_eval
from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.elasticsearch.config.config import Config
from colbert.elasticsearch.services.es_services import ElasticSearchService
from colbert.utils.runs import Run

random.seed(1234)

config_ = Config()
class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps
        self.es_service = ElasticSearchService(config_)
        self.query_tokenizer = QueryTokenizer(
            args.query_maxlen, args.pretrained_tokenizer)
        self.doc_tokenizer = DocTokenizer(
            args.doc_maxlen, args.pretrained_tokenizer)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0
        self.doc_maxlen = args.doc_maxlen
        self.query_maxlen = args.query_maxlen
        self.fine_tune = args.fine_tune

        if self.fine_tune:
            # fine_tuned dataset
            self.queries = self._load_queries(args.queries)
            self.triples = self._load_triples(args.triples, rank, nranks)
            self.qpa_pairs = self._load_qpa_pairs(args.qpa_pairs)
            self.positive_pairs = self._load_positive_pairs(
                args.positives, rank, nranks)
            self.collection = self._load_collection(args.collection)
        else:
        #     # pretrained_dataset
        #     self.queries = self._load_queries(args.queries)
        #     self.positive_collection = self._load_collection(
        #         args.positive_collection)
        #     self.negative_collection = self._load_collection(
        #         args.negative_collection)
            self.top500_path = args.top500
            self.collection_len = self.get_max_document_len()

        # self.collection_keys = list(self.collection.keys())

    def get_max_document_len(self):
        with open(self.top500_path, "r", encoding="utf8") as f:
            count = 0
            for _ in f:
                if _:
                    count += 1
        return count

    def get_random_top500_idx(self, indices):
        chosen_pid = []
        with open(self.top500_path, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                if i in indices:
                    line = literal_eval(line)
                    pid = random.sample(line, 1)[0]
                    chosen_pid.append(pid)
                    continue
        return chosen_pid

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")
        print_message(rank, nranks)
        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                print(line)
                if line_idx % nranks == rank:
                    qid, pos, neg = ujson.loads(line)
                    triples.append((qid, pos, neg))

        return triples

    def _load_qpa_pairs(self, path):
        print_message("#> Loading queries passages answer pairs...")

        qpa_dict = {}

        with open(path) as f:
            for line in f:
                pid, qid, ans, _ = line.split("\t")
                qpa_dict[(qid, pid)] = ans

        return qpa_dict

    def _load_positive_pairs(self, path, rank, nranks):
        print_message("#> Loading positive pairs...")
        positive_pairs = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pid = line.strip().split('\t')
                    positive_pairs.append((qid, pid))

        return positive_pairs

    def _load_queries(self, path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in tqdm(f):
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = word_tokenize(query, format="text")

        return queries

    def _load_collection(self, path):
        print_message("#> Loading collection...")

        collection = {}

        with open(path) as f:
            for _, line in enumerate(f):
                pid, passage = line.strip().split('\t')
                pid = int(pid)
                collection[pid] = word_tokenize(passage, format="text")

        return collection

    def _check_document_length(self, qid=0, pid=0, sentences=[], augment=False):
        if augment:
            return self._get_inner_documents(sentences)

        try:
            answer = self.qpa_pairs[(qid, pid)]
            if answer == "NO ANSWER":
                return self._get_inner_documents(sentences)
        except:
            return self._get_inner_documents(sentences)

        for idx, sentence in sentences:
            if answer in sentence:
                if idx + 1 != len(sentences) - 1 and idx - 1 != 0:
                    sentences = sentences[idx-1:idx+1]
                elif idx - 1 == 0 and idx + 1 == len(sentences) - 1:
                    sentences = sentences[idx]
                elif idx - 1 == 0:
                    sentences = sentences[:idx + 1]
                else:
                    sentences = sentences[idx-1:]
                return " ".join(sentences), sentences

    def _get_inner_documents(self, sentences):
        start_idx = random.choice(range(int(len(sentences)/2)))
        if start_idx + 2 >= len(sentences):
            end_idx = len(sentences - 1)
        else:
            end_idx = random.choice(range(start_idx + 2, len(sentences)))
        # print(start_idx, end_idx)
        sentences = sentences[start_idx:end_idx]
        pos = " ".join(sentences)
        return pos, sentences

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.positive_pairs)

    def __next__(self):
        queries, positives, negatives = [], [], []

        if self.fine_tune:
            p_augmented_query = random.uniform(0, 0.8)
            bsize = round(p_augmented_query * self.bsize)
            samples = random.sample(self.positive_pairs, bsize)

            for qid, pid_pos in samples:
                qid = int(qid)
                pid_pos = int(pid_pos)

                pid_neg = pid_pos
                while pid_neg == pid_pos:
                    pid_neg = random.choice(self.collection_keys)

                query = self.queries[qid]
                pos = self.collection[pid_pos]
                if len(pos.split()) > self.doc_maxlen:
                    pos = self._check_document_length(
                        qid, pid_pos, sent_tokenize(pos))[0]

                neg = self.collection[pid_neg]

                queries.append(query)
                positives.append(pos)
                negatives.append(neg)

            # get augmented queries
            i = 0
            # print_message("Start getting augmented queries")
            while i < self.bsize - bsize:
                # for i in range(self.bsize - bsize):
                pid_pos = random.choice(self.collection_keys)
                pos = self.collection[pid_pos]
                sentences = sent_tokenize(pos)
                if len(sentences) < 3:
                    continue

                pid_neg = pid_pos
                while pid_neg == pid_pos:
                    pid_neg = random.choice(self.collection_keys)

                neg = self.collection[pid_neg]
                if len(pos.split()) > self.doc_maxlen:
                    pos, sentences = self._check_document_length(sentences=sentences,
                                                                augment=True)

                num_sample = random.choice([1, 2])
                samples = random.sample(sentences, num_sample)

                query = " ".join(samples)
                queries.append(query)
                positives.append(pos)
                negatives.append(neg)
                i += 1
            # for query, pos, neg in zip(queries, positives, negatives):
            #     print_triples(query, pos, neg)
        else:
            qids = random.sample(range(self.collection_len), self.bsize)
            # queries = []
            # for qid in queries:
            # print(qids)
            neg_pids = self.get_random_top500_idx(qids)
            # print(neg_pids)
            
            for qid, neg_pid in zip(qids, neg_pids):
                query = self.es_service.search_question(qid)[0][1]
                pos = self.es_service.search_document(qid)[0][1]
                neg = self.es_service.search_document(neg_pid)[0][2]

                queries.append(word_tokenize(query, format="text"))
                positives.append(word_tokenize(pos, format="text"))
                negatives.append(word_tokenize(neg, format="text"))

            # for query, pos, neg in zip(queries, positives, negatives):
            #     print_triples(query, pos, neg)
        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(
            f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx

# def print_triples(query, pos, neg):
#     print(f"Query: {query}")
#     print(f"Positive passage: {pos}")
#     print(f"Negative passage: {neg}")
#     print("-"*50)
    
# if __name__ == "__main__":
#     reader = LazyBatcher(args)

#     for batch_idx, BatchSteps in zip(range(0, 2), reader):
#         print_message(f"Batch {batch_idx}")
#         for queries, passages in BatchSteps:
#             pass
