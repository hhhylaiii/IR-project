import os
from VectorSpace import VectorSpace
import numpy as np

def load_smaller_dataset(root="smaller_dataset"):
    vs = VectorSpace()
    doc_names, documents = vs.load_documents(os.path.join(root, "collections"))
    quer_names, queries = vs.load_documents(os.path.join(root, "queries"))

    relevant={}
    with open(os.path.join(root, "rel.tsv"), 'r', encoding='utf-8') as f:
        for line in f:
            qid, did_list_srt = line.strip().split('\t')
            did_list = did_list_srt.strip("[]").split(',')
            relevant[qid] = set([int(p.strip()) for p in did_list if p.strip()])
    return doc_names, documents, quer_names, queries, relevant

def build_index(documents, doc_names):
    vs = VectorSpace()
    vs.doc_names = doc_names
    vs.build(documents)
    return vs

def single_query_topk(query, index:VectorSpace, k=10, use_tfidf=True):
    topk, _, _ =index.search([query], use_tfidf=use_tfidf)
    return [int(name[1:-4]) for name, _ in topk[:k]]

def rr_at_10(pred_ids, relevant_set):
    for rank, pid in enumerate(pred_ids[:10], 1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0

def ap_at_10(pred_ids, relevant_set):
    hits = 0
    ap = 0.0
    for rank, pid in enumerate(pred_ids[:10], 1):
        if pid in relevant_set:
            hits += 1
            ap += hits / rank
    rel_nums = min(len(relevant_set), 10)
    return ap / rel_nums if rel_nums > 0 else 0.0

def recall_at_10(pred_ids, relevant_set):
    return len(set(pred_ids[:10]) & relevant_set) / len(relevant_set) if len(relevant_set) > 0 else 0.0

def evaluate_all(doc_names, documents, quer_names, queries, relevant, k=10, use_tfidf=True):
    index = build_index(documents, doc_names)
    rr_list, ap_list, rc_list = [], [], []
    for qname, query in zip(quer_names, queries):
        qid = os.path.splitext(qname)[0]
        relevant_set = relevant.get(qid, set())
        
        pred_ids = single_query_topk(query, index, k=k, use_tfidf=use_tfidf)
        rr_list.append(rr_at_10(pred_ids, relevant_set))
        ap_list.append(ap_at_10(pred_ids, relevant_set))
        rc_list.append(recall_at_10(pred_ids, relevant_set))

    return float(np.mean(rr_list)), float(np.mean(ap_list)), float(np.mean(rc_list))