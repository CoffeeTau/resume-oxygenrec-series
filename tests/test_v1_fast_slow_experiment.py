import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cache_qwen_instructions_retailrocket import (
    evidence_from_history,
    generate_reasoning_with_retries,
    load_generation_progress,
    save_generation_progress,
)
from oxygenrec.data import (
    Behavior, InteractionEvent, NextItemSample, Split, TemporalBoundaries,
)
from oxygenrec.sid import SIDRegistry
from oxygenrec.llm_reasoning import GeneratedReasoning, ReasoningGenerationError
from summarize_v1_fast_slow import load_results, summarize, write_outputs
from evaluate_v1_rank_fusion import evaluate_pair, reciprocal_rank_fusion

try:
    import torch
except ImportError:
    torch = None


class QwenHistoryEvidenceTest(unittest.TestCase):
    def test_item_anchors_use_only_prior_history(self):
        registry = SIDRegistry({"a": (1, 2, 3), "target": (7, 8, 9)})
        history = (
            InteractionEvent(10, 1, "u", "a", Behavior.VIEW),
            InteractionEvent(20, 2, "u", "a", Behavior.ADD_TO_CART),
        )
        target = InteractionEvent(30, 3, "u", "target", Behavior.TRANSACTION)
        sample = NextItemSample(Split.TRAIN, "u", history, target)
        evidence = evidence_from_history(
            sample, registry, max_recent_item_anchors=5,
            max_repeat_item_anchors=3,
        )
        serialized = json.dumps(evidence, sort_keys=True)
        self.assertIn("sid-1-2-3", serialized)
        self.assertNotIn("sid-7-8-9", serialized)
        self.assertNotIn("target", serialized)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_failed_batch_splits_and_recovers_single_cases(self):
        class FakeLLM:
            def generate(self, prompts, *, max_new_tokens, generation_seed):
                if len(prompts) > 1:
                    raise ReasoningGenerationError(
                        case_index=0, raw_text="{broken", hit_token_limit=True,
                        max_new_tokens=max_new_tokens, cause=ValueError("bad"),
                    )
                return [GeneratedReasoning(raw_text="{}", parsed={"prompt": prompts[0]})]

        stats = {
            "generation_failures": 0, "split_retries": 0,
            "single_case_retries": 0, "recovered_cases": 0,
        }
        rows = generate_reasoning_with_retries(
            FakeLLM(), ["a", "b"], max_new_tokens=8,
            retry_max_new_tokens=16, max_retries=2,
            generation_seed=17, stats=stats,
        )
        self.assertEqual([row.parsed["prompt"] for row in rows], ["a", "b"])
        self.assertEqual(stats["split_retries"], 1)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_progress_round_trip_checks_sample_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.pt"
            signature = {"sample_seed": 17}
            save_generation_progress(
                path, signature=signature, sample_keys=["train:1"],
                features=torch.ones(1, 3),
                reasoning_records=[{"instruction_text": "x"}],
                next_index=1,
                retry_stats={"generation_failures": 0},
            )
            loaded = load_generation_progress(
                path, signature=signature,
                expected_sample_keys=["train:1", "train:2"],
            )
            self.assertEqual(loaded[0], 1)
            torch.testing.assert_close(loaded[2], torch.ones(1, 3))
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                load_generation_progress(
                    path, signature={"sample_seed": 23},
                    expected_sample_keys=["train:1", "train:2"],
                )


class FastSlowSummaryTest(unittest.TestCase):
    @staticmethod
    def _record(variant, seed, hr5):
        return {
            "_path": f"seed-{seed}/{variant}/result.json",
            "variant": variant,
            "seed": seed,
            "epoch": 2,
            "selection": {"best_epoch": 2},
            "sample_seed": 17,
            "sid_registry_version": "registry-v1",
            "boundaries": {"train_end_ms": 100, "validation_end_ms": 200},
            "train_samples": 20,
            "validation_samples": 10,
            "test_samples": 10,
            "experiment_protocol": {"beam_width": 10},
            "warm_start": {"checkpoint": f"seed-{seed}/base.pt"},
            "hit_rate": {"1": 0.1, "5": hr5},
            "mrr": hr5 / 2,
            "ndcg": hr5 / 1.5,
            "legal_sid_rate": 1.0,
            "retrieval": {
                "q2i_cosine": 0.2 if "q2i" in variant else None,
                "exact_repeat_recall": 0.4 if variant.startswith("igr") else None,
                "exact_repeat_recent_recall": 0.3 if variant.startswith("igr") else None,
                "exact_repeat_random_expected_recall": 0.2 if variant.startswith("igr") else None,
                "exact_repeat_eligible": 10 if variant.startswith("igr") else 0,
            },
        }

    def test_writes_relative_gain_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, hr5 in (("base", 0.10), ("qwen_instruction", 0.12)):
                path = root / "seed-17" / variant
                path.mkdir(parents=True)
                (path / "result.json").write_text(
                    json.dumps(self._record(variant, 17, hr5)), encoding="utf-8",
                )
            summary = summarize(load_results(root), split="validation")
            write_outputs(summary, root)
            gain = summary["variants"]["qwen_instruction"][
                "hr5_relative_gain_vs_base"
            ]
            self.assertAlmostEqual(gain, 0.2)
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.csv").is_file())
            self.assertIn("qwen_instruction", (root / "summary.md").read_text())

    def test_summary_exposes_igr_delta_against_recency(self):
        grouped = {
            "base": [self._record("base", 17, 0.10)],
            "igr_qwen_q2i": [self._record("igr_qwen_q2i", 17, 0.11)],
        }
        summary = summarize(grouped, split="validation")
        retrieval = summary["variants"]["igr_qwen_q2i"]["retrieval"]
        self.assertAlmostEqual(retrieval["exact_repeat_recall"]["mean"], 0.4)
        self.assertAlmostEqual(retrieval["exact_repeat_recent_recall"]["mean"], 0.3)

    def test_rejects_mixed_sample_cohorts(self):
        grouped = {
            "base": [self._record("base", 17, 0.10)],
            "qwen_instruction": [self._record("qwen_instruction", 17, 0.12)],
        }
        grouped["qwen_instruction"][0]["sample_seed"] = 23
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            summarize(grouped, split="validation")


class FastSlowFusionTest(unittest.TestCase):
    def test_alpha_zero_preserves_base_and_positive_alpha_can_reorder(self):
        base_row = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        candidate_row = [[3, 3, 3], [1, 1, 1], [2, 2, 2]]
        self.assertEqual(
            reciprocal_rank_fusion(
                base_row, candidate_row, alpha=0.0, rrf_k=10.0,
                output_size=3,
            ),
            base_row,
        )
        fused = reciprocal_rank_fusion(
            base_row, candidate_row, alpha=1.0, rrf_k=10.0,
            output_size=3,
        )
        self.assertEqual(fused[0], [1, 1, 1])
        self.assertEqual(fused[1], [3, 3, 3])

    def test_evaluate_pair_rejects_different_cohorts(self):
        registry = SIDRegistry({"a": (1, 1, 1), "b": (2, 2, 2)})
        base = {
            "sample_keys": ["a"], "target_item_ids": ["a"],
            "semantic_ids": [[[1, 1, 1], [2, 2, 2]]],
            "beam_scores": [[0.0, -1.0]],
        }
        candidate = dict(base)
        candidate["sample_keys"] = ["different"]
        with self.assertRaisesRegex(ValueError, "cohort differ"):
            evaluate_pair(base, candidate, registry, alpha=0.5, rrf_k=10.0)


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class CommonBaseWarmStartTest(unittest.TestCase):
    def test_expands_history_positions_and_checks_protocol(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel
        from train_retailrocket import load_compatible_model_checkpoint

        boundaries = TemporalBoundaries(100, 200)
        source = OxygenRECModel(OxygenRECConfig(
            sid_width=8, hidden_size=8, attention_heads=2,
            encoder_layers=1, decoder_layers=1, feedforward_size=16,
            max_history_items=4, dropout=0.0,
        ))
        target = OxygenRECModel(OxygenRECConfig(
            sid_width=8, hidden_size=8, attention_heads=2,
            encoder_layers=1, decoder_layers=1, feedforward_size=16,
            max_history_items=4, igr_top_k=2, dropout=0.0,
            q2i_weight=0.05, q2i_decoder_weight=0.25,
            q2i_contrastive_weight=1.0,
        ))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pt"
            torch.save({
                "epoch": 3,
                "sid_registry_version": "registry-v1",
                "boundaries": {
                    "train_end_ms": 100, "validation_end_ms": 200,
                },
                "model_config": asdict(source.config),
                "model_state": source.state_dict(),
            }, checkpoint)
            report = load_compatible_model_checkpoint(
                target, checkpoint, device=torch.device("cpu"),
                registry_version="registry-v1", boundaries=boundaries,
            )
        self.assertEqual(report["source_epoch"], 3)
        self.assertEqual(report["partial_tensors"][0]["name"], "history_positions.weight")
        self.assertIsNotNone(target.query_to_decoder)
        torch.testing.assert_close(
            target.history_positions.weight[:4], source.history_positions.weight,
        )

    def test_qwen_instruction_variant_consumes_cached_features_without_igr(self):
        from oxygenrec.instruction_cache import instruction_sample_key
        from train_retailrocket import tensor_batch

        registry = SIDRegistry({"a": (1, 2, 3), "target": (4, 5, 6)})
        history = (InteractionEvent(10, 1, "u", "a", Behavior.VIEW),)
        target = InteractionEvent(20, 2, "u", "target", Behavior.VIEW)
        sample = NextItemSample(Split.TRAIN, "u", history, target)
        args = SimpleNamespace(
            variant="qwen_instruction", max_history=1, long_history=2,
            igr_top_k=1,
        )
        features = torch.tensor([[0.1, 0.2, 0.3]])
        batch = tensor_batch(
            [sample], registry, args, torch.device("cpu"),
            instruction_cache=(features, {instruction_sample_key(sample): 0}),
        )
        torch.testing.assert_close(batch["instruction_features"], features)
        self.assertEqual(tuple(batch["trigger_sids"].shape), (1, 3))
        self.assertNotIn("long_history_sids", batch)

    def test_tiny_training_selects_best_epoch_and_writes_held_out_test(self):
        import train_retailrocket

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            rows = ["timestamp,visitorid,event,itemid,transactionid"]
            for user in ("u1", "u2"):
                for source, (timestamp, item) in enumerate(
                    ((10, "a"), (20, "b"), (30, "c"), (40, "a"),
                     (50, "b"), (90, "c"), (100, "a")),
                    start=1,
                ):
                    rows.append(f"{timestamp},{user},view,{item},")
            events.write_text("\n".join(rows) + "\n", encoding="utf-8")
            registry_path = root / "registry.json"
            SIDRegistry({
                "a": (0, 0, 0), "b": (1, 1, 1), "c": (2, 2, 2),
            }).to_json(registry_path)
            output = root / "run"
            argv = [
                "train_retailrocket.py", "--events", str(events),
                "--sid-registry", str(registry_path), "--device", "cpu",
                "--variant", "base", "--matched-igr-cohort",
                "--max-history", "1", "--long-history", "2",
                "--igr-top-k", "1", "--max-train-samples", "2",
                "--max-validation-samples", "2", "--max-test-samples", "2",
                "--evaluate-test", "--batch-size", "2", "--epochs", "1",
                "--hidden-size", "8", "--attention-heads", "2",
                "--encoder-layers", "1", "--decoder-layers", "1",
                "--beam-width", "3", "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(train_retailrocket.main(), 0)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["selection"]["best_epoch"], 1)
            self.assertEqual(result["test"]["sample_count"], 2)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "validation_rankings.json").is_file())
            self.assertTrue((output / "test_rankings.json").is_file())


if __name__ == "__main__":
    unittest.main()
