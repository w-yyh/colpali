# experiments/visualize_results.py
"""
Aggregate all result JSON files and generate a comparison bar chart.

Usage:
    python experiments/visualize_results.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
from experiments.config import RESULTS_DIR

RESULT_FILES = {
    "Clean (Baseline)":       "results_clean.json",
    "Degraded: Heavy Noise":  "results_degraded_heavy_noise.json",
    "Degraded: Motion Blur":  "results_degraded_motion_blur.json",
    "Degraded: Tilt 15 deg":  "results_degraded_tilt.json",
    "Degraded: JPEG Q5":      "results_degraded_jpeg_low.json",
    "Restored: NLMeans":      "results_restored_heavy_noise_nlmeans.json",
    "Restored: Wiener":       "results_restored_heavy_noise_wiener.json",
    "Segmented":              "results_segmented.json",
}

COLOR_MAP = {
    "Baseline": "#2ecc71",
    "Degraded": "#e74c3c",
    "Restored": "#3498db",
    "Segmented": "#9b59b6",
}


def avg_ndcg(data: dict) -> float:
    vals = [v.get("ndcg@5", 0) for v in data.values() if isinstance(v, dict)]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    labels, values, colors = [], [], []
    for label, fname in RESULT_FILES.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found")
            continue
        data = json.loads(fpath.read_text())
        key = next((k for k in COLOR_MAP if k in label), "Baseline")
        labels.append(label)
        values.append(avg_ndcg(data))
        colors.append(COLOR_MAP[key])

    if not labels:
        print("No results found. Run run_benchmark.py for each condition first.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Average nDCG@5", fontsize=12)
    ax.set_title("ColPali-Robust: Retrieval Performance Under Degradation & Enhancement", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "comparison_chart.png"
    plt.savefig(out, dpi=150)
    print(f"Chart saved to {out}")


if __name__ == "__main__":
    main()
