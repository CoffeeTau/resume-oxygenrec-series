#!/usr/bin/env python3
"""Evaluate conservative Base/Slow reciprocal-rank fusion on saved beam rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.evaluation import evaluate_sid_ranking
from oxygenrec.sid import SIDRegistry


DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--candidates", nargs="+",
        default=("qwen_instruction", "qwen_q2i", "igr_qwen_q2i"),
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Use one validation-selected candidate weight, normally for confirm/test.",
    )
    parser.add_argument("--rrf-k", type=float, default=10.0)
    return parser.parse_args()


def load_rankings(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"ranking artifact not found: {path}; rerun the V1.2 profile"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("sample_keys", "target_item_ids", "semantic_ids", "beam_scores")
    if any(name not in payload for name in required):
        raise ValueError(f"invalid ranking artifact: {path}")
    row_count = len(payload["sample_keys"])
    if any(len(payload[name]) != row_count for name in required[1:]):
        raise ValueError(f"ranking artifact row counts do not match: {path}")
    return payload


def _sid_key(codes: list[int]) -> tuple[int, ...]:
    return tuple(int(code) for code in codes)


def reciprocal_rank_fusion(
    base: list[list[int]],
    candidate: list[list[int]],
    *,
    alpha: float,
    rrf_k: float,
    output_size: int,
) -> list[list[int]]:
    if alpha < 0:
        raise ValueError("alpha cannot be negative")
    if rrf_k <= 0:
        raise ValueError("rrf-k must be positive")
    scores: dict[tuple[int, ...], float] = {}
    base_rank: dict[tuple[int, ...], int] = {}
    candidate_rank: dict[tuple[int, ...], int] = {}
    for rank, codes in enumerate(base, start=1):
        key = _sid_key(codes)
        if key in base_rank:
            continue
        base_rank[key] = rank
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    for rank, codes in enumerate(candidate, start=1):
        key = _sid_key(codes)
        if key in candidate_rank:
            continue
        candidate_rank[key] = rank
        scores[key] = scores.get(key, 0.0) + alpha / (rrf_k + rank)
    ordered = sorted(
        scores,
        key=lambda key: (
            -scores[key], base_rank.get(key, 10**9),
            candidate_rank.get(key, 10**9), key,
        ),
    )
    return [list(key) for key in ordered[:output_size]]


def evaluate_pair(
    base: dict,
    candidate: dict,
    registry: SIDRegistry,
    *,
    alpha: float,
    rrf_k: float,
) -> dict:
    if base["sample_keys"] != candidate["sample_keys"]:
        raise ValueError("Base and candidate sample order/cohort differ")
    if base["target_item_ids"] != candidate["target_item_ids"]:
        raise ValueError("Base and candidate targets differ")
    output_size = min(
        min(len(row) for row in base["semantic_ids"]),
        min(len(row) for row in candidate["semantic_ids"]),
    )
    predictions = [
        reciprocal_rank_fusion(
            base_row, candidate_row, alpha=alpha, rrf_k=rrf_k,
            output_size=output_size,
        )
        for base_row, candidate_row in zip(
            base["semantic_ids"], candidate["semantic_ids"], strict=True
        )
    ]
    ks = tuple(k for k in (1, 5, 10) if k <= output_size)
    metrics = evaluate_sid_ranking(
        predictions, base["target_item_ids"], registry, ks=ks,
    )
    changed_top5 = sum(
        fused[:5] != original[:5]
        for fused, original in zip(predictions, base["semantic_ids"], strict=True)
    )
    return {
        "alpha": alpha,
        "sample_count": metrics.sample_count,
        "hit_rate": dict(metrics.hit_rate),
        "mrr": metrics.mrr,
        "ndcg": metrics.ndcg,
        "legal_sid_rate": metrics.legal_sid_rate,
        "changed_top5_fraction": changed_top5 / metrics.sample_count,
    }


def hr5(row: dict) -> float:
    return float(row["hit_rate"].get("5", row["hit_rate"].get(5, 0.0)))


def main() -> int:
    args = parse_args()
    if args.alpha is not None and args.alpha < 0:
        raise ValueError("alpha cannot be negative")
    artifact_name = f"{args.split}_rankings.json"
    seed_dirs = sorted(path for path in args.root.glob("seed-*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"no seed directories under {args.root}")
    alphas = (args.alpha,) if args.alpha is not None else DEFAULT_ALPHAS
    summary = {
        "split": args.split,
        "rrf_k": args.rrf_k,
        "selection": "validation_hr5_then_mrr_then_ndcg",
        "candidates": {},
    }
    for candidate_name in args.candidates:
        alpha_rows = []
        for alpha in alphas:
            per_seed = []
            for seed_dir in seed_dirs:
                base_dir = seed_dir / "base"
                candidate_dir = seed_dir / candidate_name
                base = load_rankings(base_dir / artifact_name)
                candidate = load_rankings(candidate_dir / artifact_name)
                registry = SIDRegistry.from_json(base_dir / "sid_registry.json")
                result = evaluate_pair(
                    base, candidate, registry, alpha=alpha, rrf_k=args.rrf_k,
                )
                result["seed"] = int(seed_dir.name.removeprefix("seed-"))
                per_seed.append(result)
            alpha_rows.append({
                "alpha": alpha,
                "per_seed": per_seed,
                "hr5_mean": statistics.fmean(hr5(row) for row in per_seed),
                "mrr_mean": statistics.fmean(row["mrr"] for row in per_seed),
                "ndcg_mean": statistics.fmean(row["ndcg"] for row in per_seed),
                "changed_top5_fraction_mean": statistics.fmean(
                    row["changed_top5_fraction"] for row in per_seed
                ),
            })
        best = max(
            alpha_rows,
            key=lambda row: (row["hr5_mean"], row["mrr_mean"], row["ndcg_mean"]),
        )
        baseline_seed_rows = []
        for seed_dir in seed_dirs:
            base_dir = seed_dir / "base"
            candidate_dir = seed_dir / candidate_name
            baseline = evaluate_pair(
                load_rankings(base_dir / artifact_name),
                load_rankings(candidate_dir / artifact_name),
                SIDRegistry.from_json(base_dir / "sid_registry.json"),
                alpha=0.0, rrf_k=args.rrf_k,
            )
            baseline["seed"] = int(seed_dir.name.removeprefix("seed-"))
            baseline_seed_rows.append(baseline)
        base_hr5_by_seed = {
            row["seed"]: hr5(row) for row in baseline_seed_rows
        }
        base_hr5 = statistics.fmean(base_hr5_by_seed.values())
        best["relative_gain_vs_base"] = (
            (best["hr5_mean"] - base_hr5) / base_hr5
            if base_hr5 is not None and base_hr5 > 0 else None
        )
        best["paired_hr5"] = [
            {
                "seed": row["seed"],
                "base": base_hr5_by_seed[row["seed"]],
                "fusion": hr5(row),
                "delta": hr5(row) - base_hr5_by_seed[row["seed"]],
            }
            for row in best["per_seed"]
        ]
        best["positive_seed_count"] = sum(
            row["delta"] > 0 for row in best["paired_hr5"]
        )
        summary["candidates"][candidate_name] = {
            "best": best,
            "grid": alpha_rows,
        }
        gain = best["relative_gain_vs_base"]
        gain_text = "n/a" if gain is None else f"{gain * 100:+.2f}%"
        paired_text = ",".join(
            f"{row['seed']}:{row['delta']:+.6f}" for row in best["paired_hr5"]
        )
        print(
            f"FUSION candidate={candidate_name} split={args.split} "
            f"alpha={best['alpha']:.2f} hr5={best['hr5_mean']:.6f} "
            f"gain_vs_base={gain_text} mrr={best['mrr_mean']:.6f} "
            f"ndcg={best['ndcg_mean']:.6f} "
            f"changed_top5={best['changed_top5_fraction_mean']:.3f} "
            f"positive_seeds={best['positive_seed_count']}/{len(best['paired_hr5'])} "
            f"seed_deltas={paired_text}"
        )
    destination = args.root / f"fusion_{args.split}_summary.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FUSION_SUMMARY={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
