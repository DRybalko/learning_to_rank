from gensim.summarization.bm25 import BM25
import numpy as np

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def calculate_scores(question, candidate_set):
    model = BM25(candidate_set)
    average_idf = sum(float(val) for val in model.idf.values()) / len(model.idf)
    scores = []
    for idx, val in enumerate(candidate_set):
        scores += [model.get_score(question, idx, average_idf)]
    return scores

"""
def convert_score_to_relevance(scores, top_k):
    scores = np.array(scores)
    probs = softmax(scores)
    if probs.shape[0] <= top_k:
        idx = max(0, probs.shape[0] - 1)
        threshold_prob = probs[np.argsort(probs)][idx]
    else:
        threshold_prob = probs[np.argsort(probs)[::-1][top_k - 1]]
    rel = np.array(probs >= threshold_prob, dtype = np.int16)
    return probs, rel
"""

def convert_score_to_relevance(scores, top_k):
    scores = np.array(scores)
    max_score = np.max(scores)
    
    if (max_score == 0):
        probs = scores
    else:
        probs = scores/max_score
     
    if probs.shape[0] <= top_k:
        idx = max(0, probs.shape[0] - 1)
        threshold_prob = probs[np.argsort(probs)][idx]
    else:
        threshold_prob = probs[np.argsort(probs)[::-1]][top_k - 1]
    rel = np.array(probs >= threshold_prob, dtype = np.int16)
    return probs, rel


def predict_relevances(qids, questions, answers, top_k = 3):
    """
    Calculates relevance probabilities and labels for question-answer pairs. 
    Parameter 'top_k' specifies the number of query-answer pairs labeled as relevant
    based on predicted probability.
    
    Returns
    -------
    list(float)
        List of relevance probabilities corresponding to questions-answers pairs.
    list(float)
        List of binary relevances corresponding to questions-answers pairs
    """
    
    qids = np.array(qids)
    questions = np.array(questions)
    answers = np.array(answers)
    unique_qids = np.unique(qids)
    
    probs_full = np.zeros(answers.shape[0])
    rel_full = np.zeros(answers.shape[0])
    for qid in unique_qids:

        index_mask = qids == qid
        question = questions[index_mask][0, :]
        candidate_set = answers[index_mask]
        
        scores = calculate_scores(question, candidate_set)
        probs, rel = convert_score_to_relevance(scores, top_k)

        probs_full[index_mask] = probs
        rel_full[index_mask] = rel

    return probs_full, rel_full

def predict_relevances_sparse(qids, questions, answers, top_k = 3):
    """
    Calculates relevance probabilities and labels for question-answer pairs. 
    Parameter 'top_k' specifies the number of query-answer pairs labeled as relevant
    based on predicted probability.
    
    Returns
    -------
    list(float)
        List of relevance probabilities corresponding to questions-answers pairs.
    list(float)
        List of binary relevances corresponding to questions-answers pairs
    """
    
    qids = np.array(qids)
    unique_qids = np.unique(qids)
    
    probs_full = np.empty(0)
    rel_full = np.empty(0)
    for qid in unique_qids:
        index_mask = qids == qid
        question = questions[index_mask][0, :].todense().tolist()[0]
        candidate_set = answers[index_mask].todense().tolist()
        
        scores = calculate_scores(question, candidate_set)
        probs, rel = convert_score_to_relevance(scores, top_k)

        probs_full = np.concatenate((probs_full, probs), axis = None)
        rel_full = np.concatenate((rel_full, rel), axis = None)
    return probs_full, rel_full