import json
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def test_delta_metrics_uses_degraded_as_reference():
    from experiments.run_restoration_analysis import delta_metrics

    delta = delta_metrics(
        baseline={"ndcg@5": 0.45, "recall@5": 0.30, "mrr": 0.60, "n_queries": 10},
        candidate={"ndcg@5": 0.48, "recall@5": 0.25, "mrr": 0.61, "n_queries": 10},
    )

    assert delta == {"ndcg@5": 0.03, "recall@5": -0.05, "mrr": 0.01}


def test_build_summary_rows_match_ppt_table_shape():
    from experiments.run_restoration_analysis import build_summary_rows

    rows = build_summary_rows(
        clean_metrics={"ndcg@5": 0.70, "recall@5": 0.60, "mrr": 0.80},
        degraded_metrics={"ndcg@5": 0.45, "recall@5": 0.40, "mrr": 0.55},
        restored_results=[
            {
                "method": "gaussian",
                "metrics": {"ndcg@5": 0.43, "recall@5": 0.39, "mrr": 0.54},
                "delta_vs_degraded": {"ndcg@5": -0.02, "recall@5": -0.01, "mrr": -0.01},
            }
        ],
    )

    assert rows == [
        {
            "condition": "clean baseline",
            "method": "clean",
            "ndcg@5": 0.70,
            "recall@5": 0.60,
            "mrr": 0.80,
            "delta_ndcg@5_vs_degraded": 0.25,
        },
        {
            "condition": "degraded PD_MB_GN_JC_LR_CS",
            "method": "degraded",
            "ndcg@5": 0.45,
            "recall@5": 0.40,
            "mrr": 0.55,
            "delta_ndcg@5_vs_degraded": 0.0,
        },
        {
            "condition": "gaussian restored",
            "method": "gaussian",
            "ndcg@5": 0.43,
            "recall@5": 0.39,
            "mrr": 0.54,
            "delta_ndcg@5_vs_degraded": -0.02,
        },
    ]


def test_save_analysis_results_names_variant(tmp_path):
    from experiments.run_restoration_analysis import save_analysis_results

    path = save_analysis_results(
        tmp_path,
        {
            "variant": "PD_MB_GN_JC_LR_CS",
            "restoration_methods": ["gaussian", "nlmeans"],
            "summary_rows": [],
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.endswith("_restoration_analysis_PD_MB_GN_JC_LR_CS.json")
    assert payload["restoration_methods"] == ["gaussian", "nlmeans"]


def test_importing_restoration_analysis_does_not_import_model_stack():
    code = (
        "import sys; "
        "import experiments.run_restoration_analysis; "
        "print('transformers' in sys.modules); "
        "print('colpali_engine.models' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(__import__("pathlib").Path(__file__).parent.parent),
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == ["False", "False"]
