#!/usr/bin/env python3
"""为真实RetailRocket样本生成并缓存论文式Qwen Instruction特征。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.data.temporal import Split, TemporalBoundaries, build_next_item_samples
from oxygenrec.instruction_cache import (
    instruction_sample_key,
    save_instruction_feature_cache,
)
from oxygenrec.llm_features import build_behavior_prompt
from oxygenrec.llm_reasoning import (
    FrozenLLMReasoningGenerator,
    ReasoningGenerationError,
    contextual_instruction_text,
)
from oxygenrec.sid import SIDRegistry


GENERATION_PROGRESS_VERSION = "oxygenrec_qwen_cache_progress_v1"


def parse_args() -> argparse.Namespace:
    """定义固定样本cohort、Qwen批量生成和缓存输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/qwen_instruction_features.pt"),
    )
    parser.add_argument(
        "--reasoning-output", type=Path,
        default=Path("data/processed/qwen_instruction_reasoning.jsonl"),
    )
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--short-history", type=int, default=20)
    parser.add_argument("--long-history", type=int, default=100)
    parser.add_argument("--igr-top-k", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-generation-retries", type=int, default=2)
    parser.add_argument("--generation-seed", type=int, default=17)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument(
        "--progress-file", type=Path, default=None,
        help="Optional resumable progress checkpoint; defaults next to --output.",
    )
    parser.add_argument("--max-recent-item-anchors", type=int, default=12)
    parser.add_argument("--max-repeat-item-anchors", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def _sid_anchor(codes) -> str:
    """把公开代理SID转成不暴露原始item ID的稳定文本锚点。"""
    return "sid-" + "-".join(str(code) for code in codes)


def evidence_from_history(
    sample, registry, *, max_recent_item_anchors: int,
    max_repeat_item_anchors: int,
) -> dict[str, object]:
    """只聚合严格早于target的历史，并加入无target的SID商品锚点。"""
    known = [
        event for event in sample.history
        if event.item_id in registry.item_to_sid
    ]
    behaviors = [event.behavior.value for event in known]
    sid_counts = Counter(registry.sid_for(event.item_id).codes for event in known)
    recent = known[-max_recent_item_anchors:]
    repeated_all = sorted(
        ((codes, count) for codes, count in sid_counts.items() if count > 1),
        key=lambda row: (-row[1], row[0]),
    )
    repeated = repeated_all[:max_repeat_item_anchors]
    return {
        "history_length": len(known),
        "behavior_counts": dict(Counter(behaviors)),
        "recent_behaviors": behaviors[-5:],
        "repeated_item_kinds": len(repeated_all),
        "recent_item_anchors": [
            f"{event.behavior.value}:{_sid_anchor(registry.sid_for(event.item_id).codes)}"
            for event in recent
        ],
        "repeated_item_anchors": [
            f"{_sid_anchor(codes)}:x{count}" for codes, count in repeated
        ],
    }


def instruction_context_anchors(evidence: dict[str, object]) -> str:
    """把严格历史中的商品锚点保留在最终编码文本，避免生成时丢失。"""
    recent = evidence["recent_item_anchors"]
    repeated = evidence["repeated_item_anchors"]
    return (
        "\n商品上下文锚点："
        f"近期={','.join(recent)}；重复={','.join(repeated) if repeated else '无'}"
    )


def generate_reasoning_with_retries(
    llm, prompts, *, max_new_tokens: int, retry_max_new_tokens: int,
    max_retries: int, generation_seed: int, stats: dict[str, int],
):
    """批失败先拆到单条；单条使用更大预算和更强约束重试。"""
    try:
        return llm.generate(
            prompts, max_new_tokens=max_new_tokens,
            generation_seed=generation_seed,
        )
    except ReasoningGenerationError as error:
        stats["generation_failures"] += 1
        print(f"stage=qwen_retry batch={len(prompts)} reason={error}")
        if len(prompts) > 1:
            stats["split_retries"] += 1
            middle = len(prompts) // 2
            left = generate_reasoning_with_retries(
                llm, prompts[:middle], max_new_tokens=max_new_tokens,
                retry_max_new_tokens=retry_max_new_tokens,
                max_retries=max_retries, generation_seed=generation_seed,
                stats=stats,
            )
            right = generate_reasoning_with_retries(
                llm, prompts[middle:], max_new_tokens=max_new_tokens,
                retry_max_new_tokens=retry_max_new_tokens,
                max_retries=max_retries,
                generation_seed=generation_seed + middle,
                stats=stats,
            )
            return left + right

        retry_prompt = (
            prompts[0]
            + "\n输出约束：立即续写一个紧凑JSON对象；不要解释、不要复述输入、"
            "不要使用Markdown；五个必需字段必须完整。"
        )
        last_error = error
        for attempt in range(1, max_retries + 1):
            stats["single_case_retries"] += 1
            try:
                recovered = llm.generate(
                    [retry_prompt],
                    max_new_tokens=retry_max_new_tokens,
                    generation_seed=generation_seed + attempt * 1_000_003,
                )
                stats["recovered_cases"] += 1
                return recovered
            except ReasoningGenerationError as retry_error:
                last_error = retry_error
                print(
                    f"stage=qwen_retry_single attempt={attempt}/{max_retries} "
                    f"reason={retry_error}"
                )
        raise RuntimeError(
            "Qwen reasoning remained invalid after isolated retries; "
            f"seed={generation_seed}; last_error={last_error}"
        ) from last_error


def save_generation_progress(
    path: Path, *, signature: dict[str, object], sample_keys,
    features, reasoning_records, next_index: int, retry_stats,
) -> None:
    """原子保存可恢复的Qwen生成进度，避免单条失败丢失全部已完成样本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "format_version": GENERATION_PROGRESS_VERSION,
        "signature": signature,
        "sample_keys": tuple(sample_keys),
        "features": features.detach().to(device="cpu"),
        "reasoning_records": list(reasoning_records),
        "next_index": next_index,
        "retry_stats": dict(retry_stats),
    }, temporary)
    temporary.replace(path)


def load_generation_progress(
    path: Path, *, signature: dict[str, object], expected_sample_keys,
):
    """加载并严格核对进度文件，防止把旧cohort续到新实验。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != GENERATION_PROGRESS_VERSION:
        raise ValueError("unsupported Qwen cache progress format")
    if payload.get("signature") != signature:
        raise ValueError("Qwen cache progress signature mismatch")
    next_index = payload.get("next_index")
    sample_keys = payload.get("sample_keys")
    features = payload.get("features")
    records = payload.get("reasoning_records")
    if not isinstance(next_index, int) or not 0 <= next_index <= len(expected_sample_keys):
        raise ValueError("Qwen cache progress next_index is invalid")
    if tuple(sample_keys or ()) != tuple(expected_sample_keys[:next_index]):
        raise ValueError("Qwen cache progress sample prefix mismatch")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("Qwen cache progress features must be Tensor[N,H]")
    if features.shape[0] != next_index:
        raise ValueError("Qwen cache progress feature count mismatch")
    if not isinstance(records, list) or len(records) != next_index:
        raise ValueError("Qwen cache progress reasoning count mismatch")
    retry_stats = payload.get("retry_stats")
    if not isinstance(retry_stats, dict):
        raise ValueError("Qwen cache progress retry_stats are missing")
    return next_index, list(sample_keys), features, records, retry_stats


def main() -> None:
    """冻结Qwen完成近线生成与编码，输出训练期只读特征缓存。"""
    args = parse_args()
    positive = (
        args.max_train_samples, args.max_validation_samples,
        args.max_test_samples,
        args.short_history, args.long_history, args.igr_top_k, args.batch_size,
        args.max_input_length, args.max_new_tokens, args.max_recent_item_anchors,
        args.max_repeat_item_anchors, args.retry_max_new_tokens,
        args.progress_every_batches,
    )
    if min(positive) < 1:
        raise ValueError("sample limits, history sizes, top-k and batch-size must be positive")
    if args.igr_top_k > args.long_history:
        raise ValueError("igr-top-k cannot exceed long-history")
    if args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("retry-max-new-tokens must be >= max-new-tokens")
    if args.max_generation_retries < 1:
        raise ValueError("max-generation-retries must be positive")
    if args.output.exists() or args.reasoning_output.exists():
        raise FileExistsError("refusing to overwrite an existing instruction cache output")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    registry = SIDRegistry.from_json(args.sid_registry)
    if checkpoint.get("sid_registry_version") != registry.version:
        raise ValueError("checkpoint and --sid-registry versions do not match")
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    samples = build_next_item_samples(
        events,
        boundaries,
        min_history=args.short_history + args.igr_top_k,
        max_history=args.short_history + args.long_history,
        max_samples_per_split={
            Split.TRAIN: args.max_train_samples,
            Split.VALIDATION: args.max_validation_samples,
            Split.TEST: args.max_test_samples,
        },
        sample_seed=args.sample_seed,
    )
    selected = [
        sample for sample in samples
        if sample.split in {Split.TRAIN, Split.VALIDATION, Split.TEST}
    ]
    split_counts = Counter(sample.split.value for sample in selected)
    if split_counts["train"] != args.max_train_samples:
        raise RuntimeError("bounded cache cohort did not fill the requested train samples")
    if split_counts["validation"] != args.max_validation_samples:
        raise RuntimeError("bounded cache cohort did not fill the requested validation samples")
    if split_counts["test"] != args.max_test_samples:
        raise RuntimeError("bounded cache cohort did not fill the requested test samples")

    progress_path = args.progress_file or args.output.with_name(
        args.output.name + ".progress.pt"
    )
    expected_sample_keys = [instruction_sample_key(sample) for sample in selected]
    progress_signature = {
        "model_directory_name": args.model_path.name,
        "sid_registry_version": registry.version,
        "boundaries": asdict(boundaries),
        "sample_seed": args.sample_seed,
        "split_counts": dict(split_counts),
        "short_history": args.short_history,
        "long_history": args.long_history,
        "igr_top_k": args.igr_top_k,
        "max_recent_item_anchors": args.max_recent_item_anchors,
        "max_repeat_item_anchors": args.max_repeat_item_anchors,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "max_input_length": args.max_input_length,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "max_generation_retries": args.max_generation_retries,
        "generation_seed": args.generation_seed,
        "decoding": "qwen_official_sampling_with_json_prefill",
    }

    llm = FrozenLLMReasoningGenerator(
        args.model_path, device=args.device, dtype=args.dtype,
        max_input_length=args.max_input_length,
    )
    sample_keys: list[str] = []
    feature_batches = []
    reasoning_records = []
    instruction_texts_seen: set[str] = set()
    retry_stats = {
        "generation_failures": 0,
        "split_retries": 0,
        "single_case_retries": 0,
        "recovered_cases": 0,
    }
    resume_index = 0
    if progress_path.is_file():
        resume_index, sample_keys, completed_features, reasoning_records, retry_stats = (
            load_generation_progress(
                progress_path, signature=progress_signature,
                expected_sample_keys=expected_sample_keys,
            )
        )
        feature_batches.append(completed_features)
        instruction_texts_seen.update(
            record["instruction_text"] for record in reasoning_records
        )
        print(
            f"stage=resume_qwen_cache completed={resume_index}/{len(selected)} "
            f"progress={progress_path}"
        )
    for start in range(resume_index, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        evidence_rows = [
            evidence_from_history(
                sample, registry,
                max_recent_item_anchors=args.max_recent_item_anchors,
                max_repeat_item_anchors=args.max_repeat_item_anchors,
            )
            for sample in batch
        ]
        prompts = [build_behavior_prompt(**evidence) for evidence in evidence_rows]
        generated = generate_reasoning_with_retries(
            llm, prompts,
            max_new_tokens=args.max_new_tokens,
            retry_max_new_tokens=args.retry_max_new_tokens,
            max_retries=args.max_generation_retries,
            generation_seed=args.generation_seed + start,
            stats=retry_stats,
        )
        instruction_texts = [
            contextual_instruction_text(output.parsed)
            + instruction_context_anchors(evidence)
            for output, evidence in zip(generated, evidence_rows, strict=True)
        ]
        encoded = llm.encode_instruction_texts(
            instruction_texts, pooling="last_token",
        )
        feature_batches.append(encoded.features.detach().to(device="cpu", dtype=torch.float16))
        for sample, evidence, output, text, token_count in zip(
            batch, evidence_rows, generated, instruction_texts,
            encoded.token_counts, strict=True,
        ):
            key = instruction_sample_key(sample)
            sample_keys.append(key)
            instruction_texts_seen.add(text)
            reasoning_records.append({
                "sample_key": key,
                "split": sample.split.value,
                "input_evidence": evidence,
                "reasoning": output.parsed,
                "instruction_text": text,
                "instruction_tokens": token_count,
                "target_excluded": True,
            })
        completed_batches = (start // args.batch_size) + 1
        if (
            completed_batches % args.progress_every_batches == 0
            or len(sample_keys) == len(selected)
        ):
            save_generation_progress(
                progress_path,
                signature=progress_signature,
                sample_keys=sample_keys,
                features=torch.cat(feature_batches, dim=0),
                reasoning_records=reasoning_records,
                next_index=len(sample_keys),
                retry_stats=retry_stats,
            )
        print(
            f"stage=cache_qwen completed={len(sample_keys)}/{len(selected)} "
            f"recovered={retry_stats['recovered_cases']}"
        )

    features = torch.cat(feature_batches, dim=0)
    save_instruction_feature_cache(
        args.output,
        sample_keys=sample_keys,
        features=features,
        metadata={
            "source": "qwen_generated_contextual_reasoning_instruction",
            "model_directory_name": args.model_path.name,
            "pooling": "last_token",
            "dtype": "float16",
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "sample_seed": args.sample_seed,
            "short_history": args.short_history,
            "long_history": args.long_history,
            "igr_top_k": args.igr_top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "retry_max_new_tokens": args.retry_max_new_tokens,
            "max_generation_retries": args.max_generation_retries,
            "generation_seed": args.generation_seed,
            "decoding": "qwen_official_sampling_with_json_prefill",
            "retry_stats": retry_stats,
            "max_recent_item_anchors": args.max_recent_item_anchors,
            "max_repeat_item_anchors": args.max_repeat_item_anchors,
            "split_counts": dict(split_counts),
        },
    )
    args.reasoning_output.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in reasoning_records
        ),
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    print(
        f"OK device={args.device} cached={len(sample_keys)} "
        f"split_counts={dict(split_counts)} feature_shape={tuple(features.shape)} "
        f"unique_instruction_rate={len(instruction_texts_seen) / len(sample_keys):.6f} "
        f"retry_stats={json.dumps(retry_stats, sort_keys=True)} "
        f"cache={args.output} reasoning={args.reasoning_output}"
    )


if __name__ == "__main__":
    main()
