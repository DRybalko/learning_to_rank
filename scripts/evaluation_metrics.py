# Implementation borrowed from https://github.com/gvishal/rank_text_cnn
# MAP and MRR metrics for learning to rank results evaluation

from collections import defaultdict
import numpy as np
from sklearn import metrics

def ap_score(cands):
    """
    Calculates average precision score for all candidates cands.
    
    Parameters
    ----------
    cands: (predicted_scores, actual_labels)
    
    """
    
    y_true, y_pred = map(list, zip(*cands))
    count = 0
    score = 0
    for i, (y_true, y_pred) in enumerate(cands):
        if y_true > 0:
            count += 1.0
            score += count / (i + 1.0)
    return score / (count + 1e-6)

def map_score(qids, labels, preds):
    """
    Computes Mean Average Precision (MAP).
    
    Parameters
    ----------
    qids: list
        Question ids
    labels: list
        True relevance labels
    pred: list 
        Predicted ranking scores
    
    
    Original Code:
    https://github.com/aseveryn/deep-qa/blob/master/run_nnet.py#L403
    """

    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert pred >= 0 and pred <= 1
        qid_2_cand[qid].append((label, pred))

    avg_precs = []
    for qid, cands in qid_2_cand.items():
        avg_prec = ap_score(sorted(cands, reverse=True, key=lambda x: x[1]))
        avg_precs.append(avg_prec)
        
    return sum(avg_precs) / len(avg_precs)


def mrr_score_qid(cands):
    
    y_true, y_pred = map(list, zip(*cands))
    for i, (y_true, y_pred) in enumerate(cands):
        if y_true > 0:
            return 1./(i + 1)
    return 0

def mrr_score(qids, labels, preds):

    """
    Computes Mean Reciprocal Rank (MRR).
    
    Parameters
    ----------
    qids: list
        Question ids
    labels: list
        True relevance labels
    pred: list 
        Predicted ranking scores
    """
        
    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert pred >= 0 and pred <= 1
        qid_2_cand[qid].append((label, pred))

    mrr_score = 0
    for qid, cands in qid_2_cand.items():
        mrr_score += mrr_score_qid(sorted(cands, reverse=True, key=lambda x: x[1]))
        
    return mrr_score/len(qid_2_cand)