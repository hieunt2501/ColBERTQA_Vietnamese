import os
import ujson
import random


# import sys
# sys.path.insert(1, "../")
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
# parser.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
# parser.add_argument('--pretrained_tokenizer', dest='pretrained_tokenizer', default="../pretrained/pretrained/phobert")
# parser.add_argument('--bsize', dest='bsize', default=32, type=int)
# parser.add_argument('--positives', dest='positives', default="../dataset/sentence/positive_pairs.tsv")
# parser.add_argument('--queries', dest='queries', default="../dataset/sentence/queries.tsv")
# parser.add_argument('--collection', dest='collection', default="../dataset/sentence/collection.tsv")
# parser.add_argument('--accum', dest='accumsteps', default=2, type=int)
# args = parser.parse_args()


from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run

random.seed(1234)

class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen, args.pretrained_tokenizer)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen, args.pretrained_tokenizer)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        # self.triples = self._load_triples(args.triples, rank, nranks)
        self.positive_pairs = self._load_positive_pairs(args.positives, rank, nranks)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)
        self.collection_keys = list(self.collection.keys())

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

    def _load_positive_pairs(self, path, rank, nranks):
        print_message("#> Loading positive pairs...")
        print_message(rank, nranks)
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
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

    def _load_collection(self, path):
        print_message("#> Loading collection...")

        collection = {}

        with open(path) as f:
            for _, line in enumerate(f):
                pid, passage = line.strip().split('\t')
                pid = int(pid)
                collection[pid] = passage
                # pid, passage, title, *_ = line.strip().split('\t')
                # assert pid == 'id' or int(pid) == line_idx

                # passage = title + ' | ' + passage
                # collection.append(passage)

        return collection

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.positive_pairs)

    def __next__(self):
        queries, positives, negatives = [], [], []
        samples = random.sample(self.positive_pairs, self.bsize)
        for qid, pid_pos in samples:
            qid = int(qid)
            pid_pos = int(pid_pos)

            pid_neg = pid_pos
            while pid_neg == pid_pos:
                pid_neg = random.choice(self.collection_keys)
            
            query = self.queries[qid]
            pos = self.collection[pid_pos]
            neg = self.collection[pid_neg]

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

            # print_triples(query, pos, neg)
        # offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        # self.position = endpos

        # if offset + self.bsize > len(self.triples):
        #     raise StopIteration

        # for position in range(offset, endpos):
        #     query, pos, neg = self.triples[position]
        #     query, pos, neg = self.queries[query], self.collection[pos], self.collection[neg]

        #     queries.append(query)
        #     positives.append(pos)
        #     negatives.append(neg)

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx


# def print_triples(query, pos, neg):
#     print(f"Query: {query}")
#     print(f"Positive passage: {pos}")
#     print(f"Negative passage: {neg}")
#     print("-"*20)
# if __name__ == "__main__":
#     reader = LazyBatcher(args)

#     for batch_idx, BatchSteps in zip(range(0, 2), reader):
#         for queries, passages in BatchSteps:
#             pass
