import os

def slow_rerank(args, query, pids, passages, sent_ref=""):
    colbert = args.colbert
    inference = args.inference

    Q = inference.queryFromText([query])

    D_ = inference.docFromText(passages, bsize=args.bsize)
    scores = colbert.score(Q, D_).cpu()

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    # ranked_passages = [passages[position] for position in ranked]
    assert len(ranked_pids) == len(set(ranked_pids))

    if sent_ref:
        # sent ref: dict of key sentence id and value is tuple of passage id and passage
        ranked_pids = [sent_ref[sid][0] for sid in ranked_pids]
        ranked_passages = [sent_ref[sid][1] for sid in ranked_pids]
    else:
        ranked_passages = [passages[position] for position in ranked]  

    return list(zip(ranked_scores, ranked_pids, ranked_passages))
