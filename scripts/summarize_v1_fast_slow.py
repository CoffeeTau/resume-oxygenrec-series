#!/usr/bin/env python3
"""汇总v1 Fast-Slow四组消融，输出终端表、JSON、CSV和Markdown。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics


CANONICAL_VARIANTS = (
    "base", "qwen_instruction", "qwen_q2i", "igr_qwen_q2i",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--expected-variants", nargs="*", default=None,
        help="Fail when any listed variant is absent; default accepts available variants.",
    )
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _hr(payload: dict, k: int) -> float:
    rates = payload["hit_rate"]
    if str(k) in rates:
        return float(rates[str(k)])
    return float(rates[k])


def _metric_payload(record: dict, split: str) -> dict:
    if split == "validation":
        return record
    test = record.get("test")
    if not isinstance(test, dict):
        raise ValueError(
            f"{record.get('variant')} seed={record.get('seed')} has no test result"
        )
    return test


def load_results(root: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.glob("**/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        variant = record.get("variant")
        seed = record.get("seed")
        if not isinstance(variant, str) or not isinstance(seed, int):
            raise ValueError(f"invalid result identity: {path}")
        record["_path"] = str(path)
        grouped[variant].append(record)
    if not grouped:
        raise FileNotFoundError(f"no result.json found under {root}")
    for variant, records in grouped.items():
        seeds = [record["seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {variant}: {seeds}")
        records.sort(key=lambda row: row["seed"])
    return dict(grouped)


def validate_protocol(grouped: dict[str, list[dict]], *, split: str) -> None:
    """拒绝把不同cohort、registry、训练口径或seed集合混成一张表。"""
    base_seeds = {row["seed"] for row in grouped.get("base", [])}
    if not base_seeds:
        raise ValueError("base result is required for protocol validation")
    for variant, records in grouped.items():
        seeds = {row["seed"] for row in records}
        if seeds != base_seeds:
            raise ValueError(
                f"seed set mismatch for {variant}: expected={sorted(base_seeds)} "
                f"actual={sorted(seeds)}"
            )
    reference = grouped["base"][0]
    fields = (
        "sample_seed", "sid_registry_version", "boundaries", "train_samples",
        "validation_samples", "test_samples", "experiment_protocol",
    )
    for variant, records in grouped.items():
        for record in records:
            mismatches = {
                field: {"expected": reference.get(field), "actual": record.get(field)}
                for field in fields if record.get(field) != reference.get(field)
            }
            if mismatches:
                raise ValueError(
                    f"protocol mismatch for {variant} seed={record['seed']}: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            if split == "test" and "test" not in record:
                raise ValueError(f"missing test result for {variant} seed={record['seed']}")
    for seed in sorted(base_seeds):
        checkpoints = {
            record.get("warm_start", {}).get("checkpoint")
            for records in grouped.values() for record in records
            if record["seed"] == seed and isinstance(record.get("warm_start"), dict)
        }
        if len(checkpoints) != 1:
            raise ValueError(
                f"variants for seed={seed} do not share one Base checkpoint: "
                f"{sorted(checkpoints)}"
            )


def summarize(grouped: dict[str, list[dict]], *, split: str) -> dict:
    validate_protocol(grouped, split=split)
    summary: dict[str, object] = {"split": split, "variants": {}}
    base_by_seed = {row["seed"]: row for row in grouped["base"]}
    ordered = [name for name in CANONICAL_VARIANTS if name in grouped]
    ordered.extend(sorted(set(grouped).difference(ordered)))
    for variant in ordered:
        records = grouped[variant]
        payloads = [_metric_payload(row, split) for row in records]
        metrics = {
            "hr1": _mean_std([_hr(row, 1) for row in payloads]),
            "hr5": _mean_std([_hr(row, 5) for row in payloads]),
            "mrr": _mean_std([float(row["mrr"]) for row in payloads]),
            "ndcg": _mean_std([float(row["ndcg"]) for row in payloads]),
            "legal_sid_rate": _mean_std([
                float(row["legal_sid_rate"]) for row in payloads
            ]),
        }
        retrieval_rows = [row.get("retrieval", {}) for row in payloads]
        optional = {}
        for key in (
            "q2i_cosine", "q2i_alignment_loss", "exact_repeat_recall",
            "exact_repeat_recent_recall", "exact_repeat_random_expected_recall",
            "exact_repeat_lift_over_random", "exact_repeat_eligible", "repeat_recall",
        ):
            values = [float(row[key]) for row in retrieval_rows if row.get(key) is not None]
            optional[key] = _mean_std(values) if values else None
        paired = []
        for record in records:
            seed = record["seed"]
            if seed not in base_by_seed:
                continue
            current = _metric_payload(record, split)
            baseline = _metric_payload(base_by_seed[seed], split)
            paired.append({
                "seed": seed,
                "hr5_delta": _hr(current, 5) - _hr(baseline, 5),
                "mrr_delta": float(current["mrr"]) - float(baseline["mrr"]),
                "ndcg_delta": float(current["ndcg"]) - float(baseline["ndcg"]),
            })
        base_hr5 = statistics.fmean([
            _hr(_metric_payload(row, split), 5) for row in grouped["base"]
        ])
        hr5_mean = metrics["hr5"]["mean"]
        relative_gain = (
            (hr5_mean - base_hr5) / base_hr5 if base_hr5 > 0 else None
        )
        summary["variants"][variant] = {
            "seeds": [row["seed"] for row in records],
            "best_epochs": [row["selection"]["best_epoch"] for row in records],
            "metrics": metrics,
            "retrieval": optional,
            "hr5_relative_gain_vs_base": relative_gain,
            "paired": paired,
            "positive_hr5_seed_count": sum(row["hr5_delta"] > 0 for row in paired),
            "nonnegative_hr5_seed_count": sum(row["hr5_delta"] >= 0 for row in paired),
            "result_paths": [row["_path"] for row in records],
        }
    return summary


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = summary["variants"]
    rows = []
    for variant, row in variants.items():
        metrics = row["metrics"]
        retrieval = row["retrieval"]
        rows.append({
            "variant": variant,
            "seeds": ",".join(str(seed) for seed in row["seeds"]),
            "best_epochs": ",".join(str(epoch) for epoch in row["best_epochs"]),
            "hr5_mean": metrics["hr5"]["mean"],
            "hr5_std": metrics["hr5"]["std"],
            "hr5_relative_gain_vs_base": row["hr5_relative_gain_vs_base"],
            "mrr_mean": metrics["mrr"]["mean"],
            "mrr_std": metrics["mrr"]["std"],
            "ndcg_mean": metrics["ndcg"]["mean"],
            "ndcg_std": metrics["ndcg"]["std"],
            "legal_sid_rate": metrics["legal_sid_rate"]["mean"],
            "q2i_cosine": (
                retrieval["q2i_cosine"]["mean"]
                if retrieval["q2i_cosine"] else None
            ),
            "exact_repeat_recall": (
                retrieval["exact_repeat_recall"]["mean"]
                if retrieval["exact_repeat_recall"] else None
            ),
            "exact_repeat_recent_recall": (
                retrieval["exact_repeat_recent_recall"]["mean"]
                if retrieval["exact_repeat_recent_recall"] else None
            ),
            "exact_repeat_random_expected_recall": (
                retrieval["exact_repeat_random_expected_recall"]["mean"]
                if retrieval["exact_repeat_random_expected_recall"] else None
            ),
            "exact_repeat_delta_vs_recent": (
                retrieval["exact_repeat_recall"]["mean"]
                - retrieval["exact_repeat_recent_recall"]["mean"]
                if retrieval["exact_repeat_recall"]
                and retrieval["exact_repeat_recent_recall"] else None
            ),
            "exact_repeat_eligible": (
                retrieval["exact_repeat_eligible"]["mean"]
                if retrieval["exact_repeat_eligible"] else None
            ),
            "positive_hr5_seed_count": row["positive_hr5_seed_count"],
            "paired_seed_count": len(row["paired"]),
        })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# V1 Fast-Slow {summary['split']} 汇总",
        "",
        "| Variant | Seeds | Best epoch | HR@5 | 相对Base | MRR | NDCG | Q2I cosine | Exact IGR | Recent | Δ vs Recent |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        relative = row["hr5_relative_gain_vs_base"]
        lines.append(
            f"| {row['variant']} | {row['seeds']} | {row['best_epochs']} | "
            f"{row['hr5_mean']:.6f}±{row['hr5_std']:.6f} | "
            f"{relative * 100:+.2f}% | " if relative is not None else
            f"| {row['variant']} | {row['seeds']} | {row['best_epochs']} | "
            f"{row['hr5_mean']:.6f}±{row['hr5_std']:.6f} | n/a | "
        )
        lines[-1] += (
            f"{row['mrr_mean']:.6f} | {row['ndcg_mean']:.6f} | "
            f"{row['q2i_cosine'] if row['q2i_cosine'] is not None else 'n/a'} | "
            f"{row['exact_repeat_recall'] if row['exact_repeat_recall'] is not None else 'n/a'} | "
            f"{row['exact_repeat_recent_recall'] if row['exact_repeat_recent_recall'] is not None else 'n/a'} | "
            f"{row['exact_repeat_delta_vs_recent'] if row['exact_repeat_delta_vs_recent'] is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "说明：screen 只用于选候选；正式结论应使用 confirm 的三随机种子 test 汇总。",
        "HR@5 相对增益为主指标，MRR/NDCG、Legal SID、Q2I cosine 与 exact-item IGR recall 为辅助证据。",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    grouped = load_results(args.root)
    expected = set(args.expected_variants or ())
    missing = expected.difference(grouped)
    if missing:
        raise RuntimeError(f"missing variants: {sorted(missing)}")
    summary = summarize(grouped, split=args.split)
    output_dir = args.output_dir or args.root
    write_outputs(summary, output_dir)
    print(f"V1_FAST_SLOW_SUMMARY split={args.split} root={args.root}")
    for variant, row in summary["variants"].items():
        metrics = row["metrics"]
        relative = row["hr5_relative_gain_vs_base"]
        relative_text = "n/a" if relative is None else f"{relative * 100:+.2f}%"
        retrieval = row["retrieval"]
        def optional_mean(name: str) -> str:
            value = retrieval.get(name)
            return "n/a" if value is None else f"{value['mean']:.6f}"
        exact = retrieval.get("exact_repeat_recall")
        recent = retrieval.get("exact_repeat_recent_recall")
        delta_recent = (
            "n/a" if exact is None or recent is None
            else f"{exact['mean'] - recent['mean']:+.6f}"
        )
        print(
            f"variant={variant} seeds={row['seeds']} "
            f"best_epochs={row['best_epochs']} "
            f"hr5={metrics['hr5']['mean']:.6f}+/-{metrics['hr5']['std']:.6f} "
            f"gain_vs_base={relative_text} mrr={metrics['mrr']['mean']:.6f} "
            f"ndcg={metrics['ndcg']['mean']:.6f} "
            f"q2i={optional_mean('q2i_cosine')} "
            f"exact_igr={optional_mean('exact_repeat_recall')} "
            f"recent={optional_mean('exact_repeat_recent_recall')} "
            f"delta_recent={delta_recent} "
            f"random={optional_mean('exact_repeat_random_expected_recall')} "
            f"positive_seeds={row['positive_hr5_seed_count']}/{len(row['paired'])}"
        )
    print(f"SUMMARY_JSON={output_dir / 'summary.json'}")
    print(f"SUMMARY_CSV={output_dir / 'summary.csv'}")
    print(f"SUMMARY_MD={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
