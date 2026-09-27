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

from cache_qwen_instructions_retailrocket import evidence_from_history
from oxygenrec.data import (
    Behavior, InteractionEvent, NextItemSample, Split, TemporalBoundaries,
)
from oxygenrec.sid import SIDRegistry
from summarize_v1_fast_slow import load_results, summarize, write_outputs

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


class FastSlowSummaryTest(unittest.TestCase):
    @staticmethod
    def _record(variant, seed, hr5):
        return {
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

    def test_rejects_mixed_sample_cohorts(self):
        grouped = {
            "base": [self._record("base", 17, 0.10)],
            "qwen_instruction": [self._record("qwen_instruction", 17, 0.12)],
        }
        grouped["qwen_instruction"][0]["sample_seed"] = 23
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            summarize(grouped, split="validation")


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


if __name__ == "__main__":
    unittest.main()
