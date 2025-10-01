#!/usr/bin/env python3
"""
CLI for grading accuracy of model responses.

Usage:
  uv run python evaluation/grader/run_grader.py single \
      --question "..." --ground-truth "..." --pred "..." [--benchmark simpleqa]

  uv run python evaluation/grader/run_grader.py batch \
      --dataset data/final_dataset/text/simpleqa_voice_episodes.json \
      --results test_output/gpt4o_simpleqa_*/gpt4o_openai_browse_batch_*.json \
      [--benchmark simpleqa]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import asyncio
from typing import Dict, Any, List, Optional

from evaluation.grader.base import GradeLabel
from evaluation.grader.llm_grader import LLMAccuracyGrader


def _load_dataset_questions(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """Map episode_id -> {question, ground_truth} from dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, Dict[str, str]] = {}
    for ep in data.get("episodes", []):
        eid = ep.get("id")
        turns = ep.get("turns", [])
        q = ""
        if turns:
            q = turns[0].get("text_content", "")
        # expected can be under turn.metadata or ep.metadata
        target = (
            (turns[0].get("metadata", {}) or {}).get("expected_answer")
            or ep.get("metadata", {}).get("expected_answer")
            or ""
        )
        if eid:
            mapping[eid] = {"question": q, "ground_truth": target}
    return mapping


def _load_results(results_glob: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(results_glob))
    episodes: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Standardized batch format stores per-episode in 'episodes' or directly as list
        eps = data.get("episodes") or data.get("results") or []
        if isinstance(eps, list) and eps:
            episodes.extend(eps)
        else:
            # Some adapters save single-episode results
            if "episode_id" in data and "turn_results" in data:
                episodes.append(data)
    return episodes


def _extract_predicted_answer(episode_result: Dict[str, Any]) -> Optional[str]:
    """Extract the assistant response text from per-episode result.

    Supports both legacy fields (turn_results/model_response) and
    standardized fields (turns/response).
    """
    # Legacy shape
    turns = episode_result.get("turn_results")
    if isinstance(turns, list) and turns:
        return turns[-1].get("model_response")

    # Standardized shape
    turns = episode_result.get("turns")
    if isinstance(turns, list) and turns:
        return turns[-1].get("response")

    return None


def _summarize_counts(labels: List[GradeLabel]) -> Dict[str, Any]:
    total = len(labels)
    c = sum(1 for l in labels if l == GradeLabel.CORRECT)
    i = sum(1 for l in labels if l == GradeLabel.INCORRECT)
    n = sum(1 for l in labels if l == GradeLabel.NOT_ATTEMPTED)
    acc = (c / total) if total else 0.0
    return {"total": total, "correct": c, "incorrect": i, "not_attempted": n, "accuracy": acc}


def main():
    # Try loading .env if available for Azure creds
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Accuracy grader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Grade a single triplet")
    p_single.add_argument("--question", required=False, default="")
    p_single.add_argument("--ground-truth", required=False, dest="ground_truth")
    p_single.add_argument("--pred", required=True)
    p_single.add_argument("--benchmark", required=False)
    # LLM-only, triad mode

    p_batch = sub.add_parser("batch", help="Grade a batch of results vs. a dataset")
    p_batch.add_argument("--dataset", required=True)
    p_batch.add_argument("--results", required=True, help="Glob to batch result JSON(s)")
    p_batch.add_argument("--benchmark", required=False)
    p_batch.add_argument("--out", required=False, help="Optional path to write detailed grades JSON")
    p_batch.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent grading requests")

    p_latest = sub.add_parser("latest", help="Auto-find latest results per model/benchmark under test_output and grade them")
    p_latest.add_argument("--models", nargs="*", default=["gpt4o", "gpt5-instant", "gpt5-thinking", "gemini-2.5-pro", "gemini-2.5-flash"], help="Models to include")
    p_latest.add_argument(
        "--benchmarks",
        nargs="*",
        default=["aime", "browsecomp", "gpqa_diamond", "mrcr", "simpleqa"],
        help="Benchmarks/datasets to include",
    )
    p_latest.add_argument("--out-dir", default="", help="Optional directory to also write an aggregate summary")
    p_latest.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent grading requests")

    args = parser.parse_args()

    grader = LLMAccuracyGrader()

    if args.cmd == "single":
        if not args.ground_truth:
            parser.error("--ground-truth is required")
        res = grader.grade(
            question=args.question,
            ground_truth=args.ground_truth,
            predicted_answer=args.pred,
            benchmark=args.benchmark,
        )
        print(json.dumps({
            "label": res.label,
            "question": args.question,
            "ground_truth": args.ground_truth,
            "extracted_final_answer": res.extracted_final_answer,
            "confidence": res.confidence,
            "reasoning": res.reasoning,
        }, default=str, indent=2))
        return 0

    if args.cmd == "batch":
        # batch
        ep_map = _load_dataset_questions(args.dataset)
        results = _load_results(args.results)

        async def _grade_all():
            sem = asyncio.Semaphore(max(1, args.max_concurrent))
            detailed_local = []
            labels_local: List[GradeLabel] = []

            async def _one(ep: Dict[str, Any]):
                eid = ep.get("episode_id")
                if not eid or eid not in ep_map:
                    return None
                qa = ep_map[eid]
                pred = _extract_predicted_answer(ep) or ""
                async with sem:
                    gres = await grader.grade_async(
                        question=qa["question"],
                        ground_truth=qa["ground_truth"],
                        predicted_answer=pred,
                        benchmark=args.benchmark,
                    )
                labels_local.append(gres.label)
                detailed_local.append({
                    "episode_id": eid,
                    "question": qa["question"],
                    "ground_truth": qa["ground_truth"],
                    "predicted_answer": pred,
                    "label": gres.label,
                    "confidence": gres.confidence,
                    "extracted_final_answer": gres.extracted_final_answer,
                })

            tasks = [
                _one(ep) for ep in results
            ]
            await asyncio.gather(*tasks)
            return labels_local, detailed_local

        labels, detailed = asyncio.run(_grade_all())

        summary = _summarize_counts(labels)
        out = {
            "summary": summary,
            "grades": detailed,
        }

        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        return 0

    if args.cmd == "latest":
        # Build dataset path resolver
        dataset_dir = Path(__file__).parent.parent.parent / 'data' / 'final_dataset' / 'text'
        def dataset_path(ds: str) -> Path:
            return dataset_dir / f"{ds}_voice_episodes.json"

        # Map model -> batch filename pattern within the run folder (new standardized adapters)
        batch_prefix = {
            'gpt4o': 'gpt4o_openai_browse_batch_',
            'gpt5-instant': 'gpt5_openai_browse_batch_',
            'gpt5-thinking': 'gpt5_openai_browse_batch_',
            'gemini-2.5-pro': 'gemini_25_pro_browse_batch_',
            'gemini-2.5-flash': 'gemini_25_flash_browse_batch_',
        }

        base = Path('test_output')
        # Also check text_output for Gemini results
        text_output_base = Path('text_output')
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        overall = []
        summary_rows = []

        for model in args.models:
            for ds in args.benchmarks:
                run_dirs = sorted(base.glob(f"{model}_{ds}_*"))

                # For Gemini models, also check text_output structure
                if model in ["gemini-2.5-pro", "gemini-2.5-flash"] and text_output_base.exists():
                    gemini_folder = "gemini_2.5_pro" if model == "gemini-2.5-pro" else "gemini_2.5_flash"
                    gemini_dir = text_output_base / gemini_folder
                    if gemini_dir.exists():
                        # Look for dataset subdirectories
                        gemini_ds_dirs = sorted(gemini_dir.glob(f"*{ds}*"))
                        run_dirs.extend(gemini_ds_dirs)

                if not run_dirs:
                    continue
                # pick most recent directory that actually contains results
                latest_dir = None
                for cand in reversed(run_dirs):
                    # any batch or per-episode results inside?
                    prefix = batch_prefix.get(model)
                    # Backward-compatible per-episode prefix patterns (legacy + current)
                    per_prefix = {
                        'gpt4o': None,
                        'gpt5-instant': None,
                        'gpt5-thinking': None,
                        'gemini-2.5-pro': 'gemini_25_pro_browse_',
                        'gemini-2.5-flash': 'gemini_25_flash_browse_',
                    }.get(model)
                    if list(cand.glob(f"{prefix}*.json")) or list(cand.glob(f"{per_prefix}*.json")):
                        latest_dir = cand
                        break
                if latest_dir is None:
                    continue
                # find batch file
                prefix = batch_prefix.get(model)
                if not prefix:
                    continue
                batch_files = sorted(latest_dir.glob(f"{prefix}*.json"))
                results_glob: Optional[str] = None
                if batch_files:
                    batch_file = str(batch_files[-1])
                    results_glob = batch_file
                else:
                    # Fallback to per-episode results if no batch file is present
                    per_prefix = {
                        'gpt4o': None,
                        'gpt5-instant': None,
                        'gpt5-thinking': None,
                        'gemini-2.5-pro': 'gemini_25_pro_browse_',
                        'gemini-2.5-flash': 'gemini_25_flash_browse_',
                    }.get(model)
                    if per_prefix:
                        per_files = sorted(latest_dir.glob(f"{per_prefix}*.json"))
                        if per_files:
                            results_glob = str(latest_dir / f"{per_prefix}*.json")
                if not results_glob:
                    continue

                ds_path = dataset_path(ds)
                if not ds_path.exists():
                    continue

                # Reuse batch grading pipeline
                ep_map = _load_dataset_questions(str(ds_path))
                results = _load_results(results_glob)

                async def _grade_all_latest():
                    sem = asyncio.Semaphore(max(1, args.max_concurrent))
                    detailed_local = []
                    labels_local: List[GradeLabel] = []

                    async def _one(ep: Dict[str, Any]):
                        eid = ep.get("episode_id")
                        if not eid or eid not in ep_map:
                            return None
                        qa = ep_map[eid]
                        pred = _extract_predicted_answer(ep) or ""
                        async with sem:
                            gres = await grader.grade_async(
                                question=qa["question"],
                                ground_truth=qa["ground_truth"],
                                predicted_answer=pred,
                                benchmark=ds,
                            )
                        labels_local.append(gres.label)
                        detailed_local.append({
                            "episode_id": eid,
                            "question": qa["question"],
                            "ground_truth": qa["ground_truth"],
                            "predicted_answer": pred,
                            "label": gres.label,
                            "confidence": gres.confidence,
                            "extracted_final_answer": gres.extracted_final_answer,
                            "model": model,
                            "dataset": ds,
                            "results_file": results_glob,
                })

                    await asyncio.gather(*[ _one(ep) for ep in results ])
                    return labels_local, detailed_local

                labels, detailed = asyncio.run(_grade_all_latest())
                overall.extend(detailed)
                summary = _summarize_counts(labels)
                results_file_for_summary = results_glob or ""
                summary_rows.append({
                    "model": model,
                    "dataset": ds,
                    **summary,
                    "results_dir": str(latest_dir),
                    "results_file": results_file_for_summary,
                })

                # write per-pair file into the corresponding run folder
                pair_out_inplace = Path(latest_dir) / "llm_grades.json"
                with open(pair_out_inplace, 'w', encoding='utf-8') as f:
                    json.dump({"summary": summary, "grades": detailed}, f, indent=2)

                # optionally also write to central out-dir if provided
                if out_dir:
                    pair_out = out_dir / f"{model}_{ds}_grades_llm.json"
                    with open(pair_out, 'w', encoding='utf-8') as f:
                        json.dump({"summary": summary, "grades": detailed}, f, indent=2)

        # write aggregate
        agg = {
            "pairs": summary_rows,
            "total_pairs": len(summary_rows),
        }
        print(json.dumps(agg, indent=2))
        if out_dir:
            with open(out_dir / "summary_latest_grades.json", 'w', encoding='utf-8') as f:
                json.dump(agg, f, indent=2)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
