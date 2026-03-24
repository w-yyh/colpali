"""Standard IR evaluation metrics: nDCG@K, Recall@K, MRR."""
import math
from typing import List, Set


def ndcg_at_k(scores: List[float], relevant: Set[int], k: int = 5) -> float:
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    dcg  = sum(1.0 / math.log2(pos + 2) for pos, doc in enumerate(ranked) if doc in relevant)
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: List[int], relevant: Set[int], k: int = 5) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(ranked: List[int], relevant: Set[int]) -> float:
    for pos, doc in enumerate(ranked):
        if doc in relevant:
            return 1.0 / (pos + 1)
    return 0.0
