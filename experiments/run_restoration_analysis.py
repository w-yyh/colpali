#!/usr/bin/env python
"""Run clean/degraded/restored retrieval analysis on the local HR subset.

Owner: Wang Yuhao.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "degraded_dataset"
DEFAULT_DOC_ID = "employment_and_social_developments_in_europe_2024-KEBD24002ENN"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "colqwen2-v1.0"
DEFAULT_VARIANT = "PD_MB_GN_JC_LR_CS"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "restoration_analysis"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "results" / "restoration_cache"
DEFAULT_RESTORATION_METHODS = ["gaussian", "nlmeans", "wiener"]
METRIC_KEYS = ("ndcg@5", "recall@5", "mrr")


def _local_benchmark_helpers():
    from experiments import run_local_hr_benchmark as local_hr

    return local_hr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean, degraded, and lightweight-restored ColQwen2 retrieval on the local HR subset."
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Degraded variant suffix to evaluate.")
    parser.add_argument("--rest", nargs="+", default=DEFAULT_RESTORATION_METHODS, help="Restoration methods to run.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None, help="Defaults to the same path as --model.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-cross-doc-queries",
        action="store_true",
        help="Keep queries that also have relevant pages in other documents.",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local Hugging Face files.")
    parser.add_argument("--list-variants", action="store_true", help="Print available degraded variants and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset wiring without loading the model.")
    return parser.parse_args()


def validate_restoration_methods(methods: Sequence[str]) -> List[str]:
    unknown = sorted(set(methods) - set(DEFAULT_RESTORATION_METHODS))
    if unknown:
        raise ValueError(f"Unknown restoration methods: {unknown}. Available: {DEFAULT_RESTORATION_METHODS}")
    return list(dict.fromkeys(methods))


def delta_metrics(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(candidate[key]) - float(baseline[key]), 10) for key in METRIC_KEYS}


def _metric_row(condition: str, method: str, metrics: Dict[str, float], delta_ndcg: float) -> Dict[str, float | str]:
    return {
        "condition": condition,
        "method": method,
        "ndcg@5": metrics["ndcg@5"],
        "recall@5": metrics["recall@5"],
        "mrr": metrics["mrr"],
        "delta_ndcg@5_vs_degraded": delta_ndcg,
    }


def build_summary_rows(
    clean_metrics: Dict[str, float],
    degraded_metrics: Dict[str, float],
    restored_results: Sequence[Dict[str, object]],
    variant: str = DEFAULT_VARIANT,
) -> List[Dict[str, float | str]]:
    rows = [
        _metric_row(
            "clean baseline",
            "clean",
            clean_metrics,
            delta_metrics(degraded_metrics, clean_metrics)["ndcg@5"],
        ),
        _metric_row(f"degraded {variant}", "degraded", degraded_metrics, 0.0),
    ]
    for result in restored_results:
        metrics = result["metrics"]
        delta = result["delta_vs_degraded"]
        rows.append(
            _metric_row(
                f"{result['method']} restored",
                str(result["method"]),
                metrics,
                delta["ndcg@5"],
            )
        )
    return rows


def save_analysis_results(output_dir: Path, payload: Dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_restoration_analysis_{payload['variant']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _score_embeddings(
    processor,
    query_embeddings: Sequence[torch.Tensor],
    doc_embeddings: Sequence[object],
    query_ids: Sequence[int],
    relevant_pages: Dict[int, set[int]],
    score_batch_size: int,
    device: str,
) -> Dict[str, float]:
    local_hr = _local_benchmark_helpers()
    scores_matrix = processor.score_multi_vector(
        list(query_embeddings),
        list(doc_embeddings),
        batch_size=score_batch_size,
        device=device,
    )
    return local_hr.compute_metrics(scores_matrix, query_ids, relevant_pages)


def _build_restoration_preprocessor(method: str):
    from experiments.invariant_embeddings import build_restoration_preprocessor

    return build_restoration_preprocessor(method)


def _encode_condition(
    model,
    processor,
    dataset_root: Path,
    doc_id: str,
    mode: str,
    variant: str,
    batch_size: int,
    device: str,
    cache_root: Path,
    max_docs: int | None,
    overwrite_cache: bool,
    preprocess_fn=None,
):
    from experiments.invariant_embeddings import load_or_encode_page_embeddings

    return load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=dataset_root,
        doc_id=doc_id,
        mode=mode,
        variant=variant,
        batch_size=batch_size,
        device=device,
        cache_root=cache_root,
        max_docs=max_docs,
        overwrite_cache=overwrite_cache,
        preprocess_fn=preprocess_fn,
    )


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    local_hr = _local_benchmark_helpers()
    restoration_methods = validate_restoration_methods(args.rest)
    tables = local_hr.load_tables(args.dataset_root)
    selected_queries, relevant_pages, query_summary = local_hr.select_queries(
        corpus_df=tables["corpus"],
        queries_df=tables["queries"],
        qrels_df=tables["qrels"],
        doc_id=args.doc_id,
        include_cross_doc_queries=args.include_cross_doc_queries,
        max_queries=args.max_queries,
    )
    clean_paths = local_hr.build_page_paths(args.dataset_root, args.doc_id, "clean", args.variant, args.max_docs)
    degraded_paths = local_hr.build_page_paths(args.dataset_root, args.doc_id, "degraded", args.variant, args.max_docs)

    payload: Dict[str, object] = {
        "experiment": "restoration_retrieval_decoupling",
        "variant": args.variant,
        "restoration_methods": restoration_methods,
        "doc_id": args.doc_id,
        "model": args.model,
        "processor": args.processor or args.model,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "include_cross_doc_queries": args.include_cross_doc_queries,
        "page_count": len(degraded_paths),
        "query_summary": query_summary,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        payload["path_checks"] = {
            "clean_first_page": str(clean_paths[0]),
            "degraded_first_page": str(degraded_paths[0]),
        }
        return payload

    import torch

    processor_name = args.processor or args.model
    model, processor = local_hr.load_model(args.model, processor_name, args.device, args.local_files_only)

    query_texts = selected_queries["query"].tolist()
    query_ids = selected_queries["query_id"].tolist()
    query_embeddings = []
    for start in range(0, len(query_texts), args.batch_size):
        batch = processor.process_queries(query_texts[start : start + args.batch_size]).to(args.device)
        with torch.no_grad():
            batch_vecs = model(**batch)
        query_embeddings.extend([vec.cpu().float() for vec in batch_vecs])

    clean_embeddings, _, clean_cache = _encode_condition(
        model,
        processor,
        args.dataset_root,
        args.doc_id,
        "clean",
        "clean",
        args.batch_size,
        args.device,
        args.cache_root,
        args.max_docs,
        args.overwrite_cache,
    )
    degraded_embeddings, _, degraded_cache = _encode_condition(
        model,
        processor,
        args.dataset_root,
        args.doc_id,
        "degraded",
        args.variant,
        args.batch_size,
        args.device,
        args.cache_root,
        args.max_docs,
        args.overwrite_cache,
    )

    clean_metrics = _score_embeddings(
        processor, query_embeddings, clean_embeddings, query_ids, relevant_pages, args.score_batch_size, args.device
    )
    degraded_metrics = _score_embeddings(
        processor, query_embeddings, degraded_embeddings, query_ids, relevant_pages, args.score_batch_size, args.device
    )

    restored_results: List[Dict[str, object]] = []
    cache_paths = {"clean": str(clean_cache), "degraded": str(degraded_cache)}
    for method in restoration_methods:
        restored_variant = f"{args.variant}__{method}"
        embeddings, _, cache_path = _encode_condition(
            model,
            processor,
            args.dataset_root,
            args.doc_id,
            "restored_proxy",
            restored_variant,
            args.batch_size,
            args.device,
            args.cache_root,
            args.max_docs,
            args.overwrite_cache,
            preprocess_fn=_build_restoration_preprocessor(method),
        )
        metrics = _score_embeddings(
            processor, query_embeddings, embeddings, query_ids, relevant_pages, args.score_batch_size, args.device
        )
        restored_results.append(
            {
                "method": method,
                "variant": restored_variant,
                "metrics": metrics,
                "delta_vs_degraded": delta_metrics(degraded_metrics, metrics),
                "cache_path": str(cache_path),
            }
        )
        cache_paths[method] = str(cache_path)

    payload["baselines"] = {
        "clean": {"metrics": clean_metrics},
        "degraded": {"variant": args.variant, "metrics": degraded_metrics},
    }
    payload["restored"] = restored_results
    payload["summary_rows"] = build_summary_rows(clean_metrics, degraded_metrics, restored_results, args.variant)
    payload["cache_paths"] = cache_paths
    return payload


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print("\n".join(_local_benchmark_helpers().list_available_variants(args.dataset_root)))
        return

    payload = run_analysis(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.dry_run:
        output_path = save_analysis_results(args.output_dir, payload)
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
