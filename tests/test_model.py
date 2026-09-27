import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class OxygenRECModelTest(unittest.TestCase):
    def setUp(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(7)
        self.model = OxygenRECModel(
            OxygenRECConfig(
                sid_width=11,
                hidden_size=16,
                attention_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                feedforward_size=32,
                dropout=0.0,
                max_history_items=4,
            )
        )
        self.history = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]]
        )
        self.padding = torch.tensor([[False, False, True], [False, False, False]])
        self.targets = torch.tensor([[1, 2, 3], [7, 8, 9]])

    def test_logits_loss_and_backward(self):
        output = self.model(
            self.history,
            self.padding,
            target_sids=self.targets,
            level_weights=(1.0, 0.7, 0.4),
        )
        self.assertEqual([tuple(item.shape) for item in output.logits], [(2, 11)] * 3)
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_q2i_diagnostics_do_not_change_loss_when_weight_is_zero(self):
        baseline = self.model(
            self.history, self.padding, target_sids=self.targets,
        )
        diagnostic = self.model(
            self.history, self.padding, target_sids=self.targets,
            compute_q2i_diagnostics=True,
        )
        torch.testing.assert_close(diagnostic.loss, baseline.loss)
        self.assertEqual(tuple(diagnostic.q2i_cosine.shape), (2,))
        self.assertIsNotNone(diagnostic.q2i_alignment_loss)

    def test_padding_codes_do_not_change_logits(self):
        self.model.eval()
        changed = self.history.clone()
        changed[0, 2] = torch.tensor([8, 8, 8])
        first = self.model(self.history, self.padding, target_sids=self.targets).logits
        second = self.model(changed, self.padding, target_sids=self.targets).logits
        for left, right in zip(first, second):
            torch.testing.assert_close(left[0], right[0])

    def test_future_target_codes_do_not_leak_into_earlier_levels(self):
        self.model.eval()
        changed = self.targets.clone()
        changed[:, 1:] = torch.tensor([[9, 10], [1, 2]])
        first = self.model(self.history, self.padding, target_sids=self.targets).logits
        second = self.model(self.history, self.padding, target_sids=changed).logits
        torch.testing.assert_close(first[0], second[0])

    def test_generation_follows_prefix_trie(self):
        from oxygenrec.sid import PrefixTrie

        self.model.eval()
        trie = PrefixTrie([(1, 2, 3), (1, 4, 5), (7, 8, 9)])
        generated = self.model.generate(self.history, self.padding, trie)
        self.assertEqual(tuple(generated.shape), (2, 3))
        for row in generated.tolist():
            self.assertTrue(trie.contains(row))

    def test_beam_search_returns_ranked_legal_paths(self):
        from oxygenrec.sid import PrefixTrie

        self.model.eval()
        trie = PrefixTrie([(1, 2, 3), (1, 4, 5), (7, 8, 9)])
        output = self.model.beam_search(
            self.history, self.padding, trie, beam_width=2
        )
        self.assertEqual(tuple(output.semantic_ids.shape), (2, 2, 3))
        self.assertEqual(tuple(output.scores.shape), (2, 2))
        for ranking, scores in zip(
            output.semantic_ids.tolist(), output.scores.tolist()
        ):
            self.assertTrue(all(trie.contains(row) for row in ranking))
            self.assertGreaterEqual(scores[0], scores[1])

    def test_history_behavior_conditioning_changes_logits(self):
        from oxygenrec.model import OxygenRECConfig, OxygenRECModel

        torch.manual_seed(11)
        model = OxygenRECModel(OxygenRECConfig(
            sid_width=11, behavior_vocab_size=3, hidden_size=16,
            attention_heads=4, encoder_layers=1, decoder_layers=1,
            feedforward_size=32, dropout=0.0, max_history_items=4,
        )).eval()
        views = torch.zeros(2, 3, dtype=torch.long)
        purchases = views.clone()
        purchases[:, 1] = 2
        first = model(
            self.history, self.padding, target_sids=self.targets,
            history_behavior_ids=views,
        ).logits
        second = model(
            self.history, self.padding, target_sids=self.targets,
            history_behavior_ids=purchases,
        ).logits
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(first, second)))

        padded_change = views.clone()
        padded_change[0, 2] = 2
        third = model(
            self.history, self.padding, target_sids=self.targets,
            history_behavior_ids=padded_change,
        ).logits
        for left, right in zip(first, third):
            torch.testing.assert_close(left[0], right[0])

    def test_sample_weighted_ntp_matches_manual_weighting(self):
        logits = (
            torch.tensor([[3.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        )
        targets = torch.tensor([[0, 0], [1, 1]])
        weights = torch.tensor([1.0, 3.0])
        loss, levels = self.model.weighted_ntp_loss(
            logits, targets, sample_weights=weights
        )
        expected_levels = []
        for level, level_logits in enumerate(logits):
            per_sample = torch.nn.functional.cross_entropy(
                level_logits, targets[:, level], reduction="none"
            )
            expected_levels.append((per_sample * weights).sum() / weights.sum())
        torch.testing.assert_close(torch.stack(levels), torch.stack(expected_levels))
        torch.testing.assert_close(loss, torch.stack(expected_levels).mean())

    def test_behavior_token_weighting_matches_paper_mean(self):
        logits = (
            torch.tensor([[3.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        )
        targets = torch.tensor([[0, 0], [1, 1]])
        token_weights = torch.tensor([[1.2, 1.2], [2.0, 2.0]])
        loss, levels = self.model.weighted_ntp_loss(
            logits, targets, token_weights=token_weights
        )
        per_token = torch.stack([
            torch.nn.functional.cross_entropy(
                level_logits, targets[:, level], reduction="none"
            )
            for level, level_logits in enumerate(logits)
        ], dim=1)
        expected = (per_token * token_weights).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            torch.stack(levels), (per_token * token_weights).mean(dim=0)
        )

    def test_sample_and_token_weights_are_mutually_exclusive(self):
        logits = tuple(torch.randn(2, 11) for _ in range(3))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.model.weighted_ntp_loss(
                logits,
                self.targets,
                sample_weights=torch.ones(2),
                token_weights=torch.ones(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
