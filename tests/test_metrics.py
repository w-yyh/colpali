# tests/test_metrics.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

def test_ndcg_perfect():
    from robust.evaluation.metrics import ndcg_at_k
    scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    assert abs(ndcg_at_k(scores, relevant={0}, k=5) - 1.0) < 1e-6

def test_ndcg_worst():
    from robust.evaluation.metrics import ndcg_at_k
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert ndcg_at_k(scores, relevant={0}, k=5) < 1.0

def test_recall_at_k():
    from robust.evaluation.metrics import recall_at_k
    ranked = [1, 3, 5, 7, 9]
    assert abs(recall_at_k(ranked, relevant={1, 3, 7}, k=3) - 2/3) < 1e-6

def test_mrr():
    from robust.evaluation.metrics import mean_reciprocal_rank
    ranked = [2, 1, 3]   # relevant doc=1 is at position 2 -> MRR=0.5
    assert abs(mean_reciprocal_rank(ranked, relevant={1}) - 0.5) < 1e-6

def test_mrr_not_found():
    from robust.evaluation.metrics import mean_reciprocal_rank
    assert mean_reciprocal_rank([1, 2, 3], relevant={99}) == 0.0
