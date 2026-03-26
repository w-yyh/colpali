#!/usr/bin/env python
"""
Segmentation experiment runner.

Runs clean baseline + adaptive segmentation on ViDoRe subsets,
saves all results, visualizations, and console log to outputs/<timestamp>/.

Usage:
    # Basic (uses config.py defaults)
    python experiments/run_segmentation_experiment.py

    # Override device / model
    python experiments/run_segmentation_experiment.py --device cuda:1 --model vidore/colqwen2-v1.0

    # Background-safe (survives terminal close)
    nohup python experiments/run_segmentation_experiment.py --device cuda:1 \
        > outputs/experiment_console.log 2>&1 &
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from experiments.config import (
    MODEL_NAME, PROCESSOR_NAME, BATCH_SIZE, VIDORE_SUBSETS, RESULTS_DIR,
)
from colpali_engine.models import ColQwen2, ColQwen2Processor
from robust.evaluation.metrics import ndcg_at_k, recall_at_k, mean_reciprocal_rank
from robust.segmentation.adaptive_seg import adaptive_segment


# ──────────────────────────────── Helpers ───────────────────────────────

def load_model(model_name, processor_name, device):
    print(f"Loading {model_name} on {device}...")
    model = ColQwen2.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(processor_name)
    return model, processor


def evaluate_subset(model, processor, subset_name, preprocess_fn, device):
    """Evaluate a single dataset subset and return metrics dict."""
    print(f"  Loading {subset_name}...")
    ds = load_dataset(subset_name, split="test")
    queries = [row["query"] for row in ds]
    images = [row["image"].convert("RGB") for row in ds]
    n = len(queries)

    # Encode queries
    all_q_vecs = []
    for i in range(0, n, BATCH_SIZE):
        inputs = processor.process_queries(queries[i:i + BATCH_SIZE]).to(device)
        with torch.no_grad():
            vecs = model(**inputs)
        all_q_vecs.extend([v.cpu().float() for v in vecs])

    # Encode documents with preprocessing
    all_d_vecs = []
    seg_times = []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="  Encoding docs"):
        batch_imgs = images[i:i + BATCH_SIZE]
        t0 = time.time()
        preprocessed = preprocess_fn(batch_imgs)
        seg_times.append(time.time() - t0)
        inputs = processor.process_images(preprocessed).to(device)
        with torch.no_grad():
            vecs = model(**inputs)
        all_d_vecs.extend([v.cpu().float() for v in vecs])

    # Scoring (pass as list — score_multi_vector handles variable-length sequences)
    scores_matrix = processor.score_multi_vector(all_q_vecs, all_d_vecs)

    # Metrics
    ndcg_list, rec_list, mrr_list = [], [], []
    for i in range(n):
        s = scores_matrix[i].tolist()
        ranked = sorted(range(n), key=lambda j: s[j], reverse=True)
        ndcg_list.append(ndcg_at_k(s, {i}, k=5))
        rec_list.append(recall_at_k(ranked, {i}, k=5))
        mrr_list.append(mean_reciprocal_rank(ranked, {i}))

    avg_seg_time = sum(seg_times) / len(seg_times) if seg_times else 0
    return {
        "ndcg@5": round(sum(ndcg_list) / n, 4),
        "recall@5": round(sum(rec_list) / n, 4),
        "mrr": round(sum(mrr_list) / n, 4),
        "n_samples": n,
        "avg_preprocess_time_per_batch_sec": round(avg_seg_time, 4),
    }


def save_sample_visualizations(subset_name, output_dir, num_samples=5):
    """Save before/after visualization for a few samples."""
    print(f"  Saving sample visualizations for {subset_name}...")
    ds = load_dataset(subset_name, split="test")
    vis_dir = output_dir / "visualizations" / subset_name.split("/")[-1]
    vis_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx in range(min(num_samples, len(ds))):
        img = ds[idx]["image"].convert("RGB")
        adaptive_result = adaptive_segment(img)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(np.array(img))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(np.array(adaptive_result))
        axes[1].set_title("Adaptive Segmentation")
        axes[1].axis("off")

        fig.suptitle(f"Sample {idx}: {ds[idx]['query'][:80]}...", fontsize=10)
        fig.tight_layout()
        fig.savefig(vis_dir / f"sample_{idx:03d}.png", dpi=150)
        plt.close(fig)


# ─────────────────────────── Main experiment ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run segmentation experiments")
    parser.add_argument("--device", default=None,
                        help="CUDA device (e.g. cuda:0, cuda:1). Defaults to config.py DEVICE.")
    parser.add_argument("--model", default=None,
                        help="Model name or local path. Defaults to config.py MODEL_NAME.")
    parser.add_argument("--processor", default=None,
                        help="Processor name or local path. Defaults to --model value.")
    parser.add_argument("--subsets", nargs="+", default=VIDORE_SUBSETS)
    args = parser.parse_args()

    from experiments.config import DEVICE as DEFAULT_DEVICE
    device = args.device or DEFAULT_DEVICE
    model_name = args.model or MODEL_NAME
    processor_name = args.processor or args.model or PROCESSOR_NAME

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"={'=' * 69}")
    print(f"  Segmentation Experiment — {timestamp}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Model:  {model_name}")
    print(f"={'=' * 69}")

    model, processor = load_model(model_name, processor_name, device)

    conditions = {
        "clean": lambda imgs: imgs,
        "segmented_adaptive": lambda imgs: [adaptive_segment(img) for img in imgs],
    }

    all_results = {}
    experiment_log = {
        "timestamp": timestamp,
        "device": device,
        "model": model_name,
        "batch_size": BATCH_SIZE,
        "subsets": args.subsets,
        "conditions": list(conditions.keys()),
    }

    for cond_name, preprocess_fn in conditions.items():
        print(f"\n{'─' * 50}")
        print(f"  Condition: {cond_name}")
        print(f"{'─' * 50}")

        cond_results = {}
        for subset in args.subsets:
            short_name = subset.split("/")[-1]
            print(f"\n  Evaluating: {short_name}")
            metrics = evaluate_subset(model, processor, subset, preprocess_fn, device)
            cond_results[short_name] = metrics
            print(f"    nDCG@5={metrics['ndcg@5']:.4f}  "
                  f"Recall@5={metrics['recall@5']:.4f}  "
                  f"MRR={metrics['mrr']:.4f}  "
                  f"Preprocess={metrics['avg_preprocess_time_per_batch_sec']:.4f}s/batch")

        all_results[cond_name] = cond_results

        # Also save per-condition to RESULTS_DIR for compatibility
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"results_{cond_name}.json").write_text(
            json.dumps(cond_results, indent=2))

    # ─── Save to timestamped output dir ───
    (output_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    (output_dir / "experiment_log.json").write_text(json.dumps(experiment_log, indent=2))

    # ─── Print & save summary ───
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    summary_lines = []
    header = f"{'Condition':<25} {'Subset':<30} {'nDCG@5':>8} {'Recall@5':>10} {'MRR':>8}"
    summary_lines.append(header)
    summary_lines.append("─" * len(header))
    print(header)
    print("─" * len(header))

    for cond_name, cond_results in all_results.items():
        for subset_name, metrics in cond_results.items():
            line = (f"{cond_name:<25} {subset_name:<30} "
                    f"{metrics['ndcg@5']:>8.4f} {metrics['recall@5']:>10.4f} {metrics['mrr']:>8.4f}")
            summary_lines.append(line)
            print(line)

    (output_dir / "summary.txt").write_text("\n".join(summary_lines))

    # ─── Generate comparison chart ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        subsets = list(next(iter(all_results.values())).keys())
        cond_names = list(all_results.keys())
        x = np.arange(len(subsets))
        width = 0.8 / len(cond_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, cond in enumerate(cond_names):
            values = [all_results[cond][s]["ndcg@5"] for s in subsets]
            bars = ax.bar(x + i * width, values, width, label=cond)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("nDCG@5")
        ax.set_title(f"Segmentation Comparison — {timestamp}")
        ax.set_xticks(x + width * (len(cond_names) - 1) / 2)
        ax.set_xticklabels(subsets, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_chart.png", dpi=150)
        plt.close(fig)
        print(f"\nChart saved: {output_dir / 'comparison_chart.png'}")
    except Exception as e:
        print(f"\nWarning: Failed to generate chart: {e}")

    # ─── Sample visualizations ───
    for subset in args.subsets:
        try:
            save_sample_visualizations(subset, output_dir)
        except Exception as e:
            print(f"Warning: Failed to save visualizations for {subset}: {e}")

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
