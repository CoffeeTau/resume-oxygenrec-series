import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.llm_features import build_behavior_prompt

try:
    import torch
except ImportError:
    torch = None


class LLMFeaturePromptTest(unittest.TestCase):
    def test_prompt_contains_only_supplied_prior_evidence(self):
        prompt = build_behavior_prompt(
            history_length=12,
            behavior_counts={"view": 10, "addtocart": 2},
            recent_behaviors=("view", "addtocart"),
            repeated_item_kinds=3,
            recent_item_anchors=("view:sid-1-2-3", "addtocart:sid-4-5-6"),
            repeated_item_anchors=("sid-1-2-3:x3",),
        )
        self.assertIn("历史长度: 12", prompt)
        self.assertIn("view=10", prompt)
        self.assertIn("不得猜测下一次", prompt)
        self.assertIn("view:sid-1-2-3", prompt)
        self.assertIn("sid-1-2-3:x3", prompt)
        self.assertNotIn("目标SID", prompt)

    def test_prompt_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            build_behavior_prompt(
                history_length=0, behavior_counts={}, recent_behaviors=(),
                repeated_item_kinds=0,
            )

    def test_pooling_contract_is_documented(self):
        from oxygenrec.llm_features import FrozenLLMInstructionEncoder

        self.assertIn("pooling", FrozenLLMInstructionEncoder.encode.__annotations__)


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class FrozenFeatureTensorBoundaryTest(unittest.TestCase):
    def test_clone_outside_inference_mode_can_feed_trainable_adapter(self):
        with torch.inference_mode():
            inference_feature = torch.ones(2, 4)
        ordinary_feature = inference_feature.clone()
        adapter = torch.nn.Linear(4, 3)
        adapter(ordinary_feature).sum().backward()
        self.assertIsNotNone(adapter.weight.grad)


if __name__ == "__main__":
    unittest.main()
