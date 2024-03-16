import numpy as np

def ndcg_at_k(recommended_list, test_list, k):
    if len(recommended_list) > k:
        recommended_list = recommended_list[:k]
    dcg = 0.0
    for i in range(k):
        if recommended_list[i-1] in test_list:
            dcg += 1 / np.log2(i+2)
    idcg = 0.0
    for i in range(min(k, len(test_list))):
        idcg += 1 / np.log2(i+2)
    return dcg / idcg if idcg > 0 else 0

def f1_at_k(recommended_list, test_list, k):
    if len(recommended_list) > k:
        recommended_list = recommended_list[:k]
    precision = len(set(recommended_list) & set(test_list)) / k
    recall = len(set(recommended_list) & set(test_list)) / len(test_list)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0