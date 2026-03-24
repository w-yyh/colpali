# experiments/pso_optimize.py
"""
PSO: find optimal NLMeans+Gaussian restoration parameters that maximize nDCG@5.

Usage:
    python experiments/pso_optimize.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import json
import numpy as np
import torch
from datasets import load_dataset

from experiments.config import MODEL_NAME, PROCESSOR_NAME, DEVICE, RESULTS_DIR, VIDORE_SUBSETS
from colpali_engine.models import ColQwen2, ColQwen2Processor
from robust.degradation.pipeline import DegradationPipeline
from robust.restoration.pipeline import RestorationPipeline
from robust.evaluation.metrics import ndcg_at_k
from robust.optimization.pso import PSOptimizer

N_EVAL = 30   # use first N samples for fast PSO objective evaluation


def make_objective(model, processor, images, queries):
    """Create an objective function that evaluates a restoration config."""
    deg = DegradationPipeline([("gaussian_noise", {"std": 30})])
    degraded = [deg(img) for img in images]

    def objective(params):
        h, sigma = max(1.0, float(params[0])), max(0.1, float(params[1]))
        rest = RestorationPipeline([
            ("nlmeans", {"h": h}),
            ("gaussian", {"sigma": sigma}),
        ])
        restored = [rest(img) for img in degraded]

        inputs = processor.process_images(restored).to(DEVICE)
        with torch.no_grad():
            d_vecs = model(**inputs)

        q_inputs = processor.process_queries(queries).to(DEVICE)
        with torch.no_grad():
            q_vecs = model(**q_inputs)

        scores_matrix = processor.score_multi_vector(q_vecs, d_vecs)
        n = len(queries)
        return sum(ndcg_at_k(scores_matrix[i].tolist(), {i}) for i in range(n)) / n

    return objective


def main():
    print(f"Loading model {MODEL_NAME}...")
    model = ColQwen2.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()
    processor = ColQwen2Processor.from_pretrained(PROCESSOR_NAME)

    subset = VIDORE_SUBSETS[0]
    print(f"Loading {N_EVAL} samples from {subset}...")
    ds = load_dataset(subset, split="test").select(range(N_EVAL))
    images  = [row["image"].convert("RGB") for row in ds]
    queries = [row["query"] for row in ds]

    print(f"Running PSO optimization...")
    objective = make_objective(model, processor, images, queries)
    optimizer = PSOptimizer(bounds=[(1, 30), (0.1, 5)], n_particles=8, iters=15)
    best_params, best_score = optimizer.optimize(objective)

    print(f"\nBest: nlmeans_h={best_params[0]:.2f}, gaussian_sigma={best_params[1]:.2f}")
    print(f"Best nDCG@5: {best_score:.4f}")

    result = {
        "nlmeans_h":      float(best_params[0]),
        "gaussian_sigma": float(best_params[1]),
        "best_ndcg@5":    best_score,
    }
    out = RESULTS_DIR / "results_pso.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
