import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oxygenrec.llm_reasoning import (
    FrozenLLMReasoningGenerator,
    contextual_instruction_text,
    parse_reasoning_json,
)

try:
    import torch
except ImportError:
    torch = None


class ReasoningJSONTest(unittest.TestCase):
    def test_parses_required_schema_with_surrounding_text(self):
        parsed = parse_reasoning_json(
            'prefix {"intent":"复购", "evidence":["购买=1"], '
            '"retrieval_strategy":"检索重复商品", '
            '"retrieval_plan":{"priority_behaviors":["transaction"],'
            '"recency":"balanced","prefer_repeated_items":true,"diversity":"low"},'
            '"constraints":["不猜目标"]} suffix'
        )
        self.assertEqual(parsed["intent"], "复购")

    def test_rejects_missing_or_wrong_fields(self):
        with self.assertRaises(ValueError):
            parse_reasoning_json('{"intent":"x"}')
        with self.assertRaises(ValueError):
            parse_reasoning_json(
                '{"intent":"x","evidence":"bad",'
                '"retrieval_strategy":"y","retrieval_plan":{},"constraints":[]}'
            )

    def test_reports_truncated_nested_json_as_invalid(self):
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            parse_reasoning_json(
                '{"intent":"x","evidence":["view=41"],'
                '"retrieval_strategy":"y","retrieval_plan":'
                '{"priority_behaviors":["view"]}'
            )

    def test_ignores_text_after_first_complete_object(self):
        parsed = parse_reasoning_json(
            '{"intent":"x","evidence":["view=41"],'
            '"retrieval_strategy":"y",'
            '"retrieval_plan":{"priority_behaviors":["view"],'
            '"recency":"recent","prefer_repeated_items":false,'
            '"diversity":"high"},"constraints":["不猜目标"]}'
            ' trailing {not json}'
        )
        self.assertEqual(parsed["retrieval_plan"]["recency"], "recent")

    def test_paper_instruction_text_does_not_consume_agentic_plan(self):
        parsed = parse_reasoning_json(
            '{"intent":"识别长期兴趣","evidence":["view=41","重复商品=2"],'
            '"retrieval_strategy":"检索历史兴趣商品",'
            '"retrieval_plan":{"priority_behaviors":["view"],'
            '"recency":"recent","prefer_repeated_items":false,'
            '"diversity":"high"},"constraints":["不猜目标"]}'
        )
        first = contextual_instruction_text(parsed)
        parsed["retrieval_plan"]["recency"] = "long_term"
        parsed["retrieval_plan"]["diversity"] = "low"
        second = contextual_instruction_text(parsed)
        self.assertEqual(first, second)
        self.assertIn("当前意图：识别长期兴趣", first)
        self.assertIn("推理依据：view=41；重复商品=2", first)
        self.assertNotIn("recent", first)


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class QwenGenerationProtocolTest(unittest.TestCase):
    def test_uses_json_prefill_and_official_sampling_parameters(self):
        valid_continuation = (
            '"intent":"复购","evidence":["购买=1"],'
            '"retrieval_strategy":"检索重复商品",'
            '"retrieval_plan":{"priority_behaviors":["transaction"],'
            '"recency":"balanced","prefer_repeated_items":true,'
            '"diversity":"low"},"constraints":["不猜目标"]}'
        )

        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return "rendered"

            def __call__(self, rendered, **kwargs):
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            def decode(self, ids, **kwargs):
                return valid_continuation

        class FakeModel:
            def __init__(self):
                self.generation_config = SimpleNamespace(
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                )
                self.seen_config = None

            def generate(self, **kwargs):
                self.seen_config = kwargs["generation_config"]
                return torch.tensor([[10, 11, 12, 13]])

        generator = object.__new__(FrozenLLMReasoningGenerator)
        generator.device = torch.device("cpu")
        generator.max_input_length = 32
        generator.tokenizer = FakeTokenizer()
        generator.model = FakeModel()
        result = generator.generate(["history"], max_new_tokens=8, generation_seed=7)

        messages, template_kwargs = generator.tokenizer.calls[0]
        self.assertEqual(messages[-1], {"role": "assistant", "content": "{"})
        self.assertTrue(template_kwargs["continue_final_message"])
        self.assertFalse(template_kwargs["enable_thinking"])
        self.assertTrue(generator.model.seen_config.do_sample)
        self.assertEqual(generator.model.seen_config.temperature, 0.7)
        self.assertEqual(result[0].parsed["intent"], "复购")
        self.assertFalse(generator.model.generation_config.do_sample)


if __name__ == "__main__":
    unittest.main()
