import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class InstructionQ2IIGRTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(17)
        self.model = OxygenRECModel(OxygenRECConfig(
            sid_width=16, hidden_size=16, attention_heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_size=32,
            dropout=0.0, max_history_items=3, scenario_vocab_size=2,
            instruction_feature_size=6, q2i_dimension=8,
            q2i_weight=0.2, igr_top_k=2,
            use_history_context_instruction=True,
            history_context_pooling="attention",
        ))
        self.short = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
        self.short_mask = torch.zeros(2, 2, dtype=torch.bool)
        self.long = torch.tensor([
            [[7, 8, 9], [3, 4, 5], [6, 7, 8], [0, 0, 0]],
            [[8, 9, 10], [1, 3, 5], [4, 6, 8], [0, 0, 0]],
        ])
        self.long_mask = torch.tensor([[False, False, False, True]] * 2)
        self.targets = torch.tensor([[7, 8, 9], [8, 9, 10]])
        self.features = torch.randn(2, 6)

    def test_joint_loss_backward_and_masked_igr(self):
        output = self.model(
            self.short, self.short_mask, target_sids=self.targets,
            scenario_ids=torch.tensor([0, 1]), instruction_features=self.features,
            long_history_sids=self.long,
            long_history_padding_mask=self.long_mask,
        )
        self.assertIsNotNone(output.q2i_loss)
        self.assertEqual(tuple(output.q2i_cosine.shape), (2,))
        torch.testing.assert_close(
            output.q2i_alignment_loss, -output.q2i_cosine.mean()
        )
        self.assertEqual(tuple(output.igr_indices.shape), (2, 2))
        self.assertTrue((output.igr_indices < 3).all())
        self.assertTrue((output.igr_scores[:, :-1] >= output.igr_scores[:, 1:]).all())
        torch.testing.assert_close(
            output.loss, output.ntp_loss + 0.2 * output.q2i_loss
        )
        output.loss.backward()
        self.assertIsNotNone(self.model.query_adapter[0].weight.grad)

    def test_q2i_auxiliary_loss_does_not_move_shared_sid_embeddings(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        model = OxygenRECModel(OxygenRECConfig(
            sid_width=16, hidden_size=16, attention_heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_size=32,
            dropout=0.0, max_history_items=3, scenario_vocab_size=2,
            instruction_feature_size=6, q2i_dimension=8, q2i_weight=0.2,
        ))
        output = model(
            self.short, self.short_mask, target_sids=self.targets,
            scenario_ids=torch.tensor([0, 1]), instruction_features=self.features,
        )
        output.q2i_loss.backward()
        self.assertIsNone(model.sid_embeddings[0].weight.grad)
        self.assertIsNotNone(model.item_adapter[0].weight.grad)

    def test_v11_query_is_coupled_to_decoder_and_uses_contrastive_loss(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(23)
        model = OxygenRECModel(OxygenRECConfig(
            sid_width=16, hidden_size=16, attention_heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_size=32,
            dropout=0.0, max_history_items=3, scenario_vocab_size=2,
            instruction_feature_size=6, q2i_dimension=8, q2i_weight=0.05,
            q2i_decoder_weight=0.25, q2i_contrastive_weight=1.0,
            q2i_temperature=0.1,
        ))
        output = model(
            self.short, self.short_mask, target_sids=self.targets,
            scenario_ids=torch.tensor([0, 1]), instruction_features=self.features,
        )
        self.assertIsNotNone(output.q2i_contrastive_loss)
        self.assertTrue(torch.isfinite(output.q2i_contrastive_loss))
        output.loss.backward()
        self.assertIsNotNone(model.query_to_decoder.weight.grad)

    def test_recency_weight_one_selects_most_recent_valid_long_items(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        model = OxygenRECModel(OxygenRECConfig(
            sid_width=16, hidden_size=16, attention_heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_size=32,
            dropout=0.0, max_history_items=3, q2i_dimension=8,
            igr_top_k=2, igr_recency_weight=1.0,
        ))
        query = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        _, _, indices, _ = model._augment_history(
            self.short, self.short_mask, self.long, self.long_mask, query,
        )
        self.assertEqual(indices.tolist(), [[2, 1], [2, 1]])

    def test_dense_instruction_changes_predictions(self):
        self.model.eval()
        zeros = torch.zeros_like(self.features)
        ones = torch.ones_like(self.features)
        first = self.model(self.short, self.short_mask, instruction_features=zeros).logits
        second = self.model(self.short, self.short_mask, instruction_features=ones).logits
        self.assertFalse(torch.allclose(first[0], second[0]))

    def test_history_context_ignores_padding_codes(self):
        self.model.eval()
        padding = self.short_mask.clone()
        padding[0, 1] = True
        changed = self.short.clone()
        changed[0, 1] = torch.tensor([13, 14, 15])
        first = self.model(self.short, padding, target_sids=self.targets).logits
        second = self.model(changed, padding, target_sids=self.targets).logits
        for left, right in zip(first, second):
            torch.testing.assert_close(left[0], right[0])

    def test_executable_plan_runs_inside_model_igr(self):
        from oxygenrec.retrieval_planning import compile_retrieval_plan

        plans = [compile_retrieval_plan(
            {"priority_behaviors": ["transaction", "view"], "recency": "balanced",
             "prefer_repeated_items": True, "diversity": "high"},
            {"view": 2, "transaction": 1},
        )] * 2
        output = self.model(
            self.short, self.short_mask, instruction_features=self.features,
            long_history_sids=self.long, long_history_padding_mask=self.long_mask,
            long_history_behavior_ids=torch.tensor([[0, 2, 0, 0]] * 2),
            retrieval_plans=plans,
            retrieval_mode="agentic_plan",
        )
        self.assertEqual(tuple(output.igr_indices.shape), (2, 2))
        self.assertTrue((output.igr_indices < 3).all())

    def test_executable_plan_reaches_generate_and_beam_search(self):
        from unittest.mock import patch

        import oxygenrec.model as model_module
        from oxygenrec.retrieval_planning import compile_retrieval_plan
        from oxygenrec.sid import PrefixTrie

        plans = [compile_retrieval_plan(
            {"priority_behaviors": ["transaction"], "recency": "balanced",
             "prefer_repeated_items": False, "diversity": "low"},
            {"view": 2, "transaction": 1},
        )] * 2
        long_behaviors = torch.tensor([[0, 2, 0, 0]] * 2)
        trie = PrefixTrie([(1, 2, 3), (7, 8, 9), (8, 9, 10)])
        shared = dict(
            instruction_features=self.features,
            long_history_sids=self.long,
            long_history_padding_mask=self.long_mask,
            long_history_behavior_ids=long_behaviors,
            retrieval_plans=plans,
            retrieval_mode="agentic_plan",
        )
        with patch.object(
            model_module, "execute_retrieval_plan",
            wraps=model_module.execute_retrieval_plan,
        ) as execute:
            generated = self.model.generate(
                self.short, self.short_mask, trie, **shared,
            )
            beams = self.model.beam_search(
                self.short, self.short_mask, trie, beam_width=2, **shared,
            )
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(tuple(generated.shape), (2, 3))
        self.assertEqual(tuple(beams.semantic_ids.shape), (2, 2, 3))

    def test_retrieval_mode_keeps_paper_and_agentic_paths_explicit(self):
        from oxygenrec.retrieval_planning import compile_retrieval_plan

        plans = [compile_retrieval_plan(
            {"priority_behaviors": ["view"], "recency": "balanced",
             "prefer_repeated_items": False, "diversity": "low"},
            {"view": 3},
        )] * 2
        shared = dict(
            instruction_features=self.features,
            long_history_sids=self.long,
            long_history_padding_mask=self.long_mask,
            long_history_behavior_ids=torch.tensor([[0, 0, 0, 0]] * 2),
        )
        paper = self.model(
            self.short, self.short_mask, retrieval_mode="paper_igr", **shared,
        )
        agentic = self.model(
            self.short, self.short_mask, retrieval_mode="agentic_plan",
            retrieval_plans=plans, **shared,
        )
        self.assertEqual(tuple(paper.igr_indices.shape), (2, 2))
        self.assertEqual(tuple(agentic.igr_indices.shape), (2, 2))
        with self.assertRaisesRegex(ValueError, "does not consume"):
            self.model(
                self.short, self.short_mask, retrieval_mode="paper_igr",
                retrieval_plans=plans, **shared,
            )
        with self.assertRaisesRegex(ValueError, "requires retrieval_plans"):
            self.model(
                self.short, self.short_mask, retrieval_mode="agentic_plan", **shared,
            )


if __name__ == "__main__":
    unittest.main()
