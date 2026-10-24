import tempfile
import unittest
from pathlib import Path

import numpy as np

from zerogpt.gpt import GPTConfig, GPTModel
from zerogpt.optim import AdamW
from scripts.train_gpt import iter_evaluation_batches, sample
from zerogpt.tokenizer import CharacterTokenizer


class GPTModelTests(unittest.TestCase):
    def small_model(self, seed: int = 4) -> GPTModel:
        return GPTModel(
            GPTConfig(
                vocab_size=5,
                context_length=6,
                num_layers=1,
                num_heads=1,
                model_width=8,
                mlp_width=16,
            ),
            seed=seed,
        )

    def test_forward_has_expected_shape_and_causal_mask(self) -> None:
        model = self.small_model()
        inputs = np.array([[0, 1, 2, 3]], dtype=np.int64)
        original = model.forward(inputs).data
        changed = inputs.copy()
        changed[0, 3] = 4
        updated = model.forward(changed).data

        self.assertEqual(original.shape, (1, 4, 5))
        np.testing.assert_array_equal(original[:, :3], updated[:, :3])
        self.assertFalse(np.array_equal(original[:, 3], updated[:, 3]))

    def test_checkpoint_round_trip_preserves_logits(self) -> None:
        model = self.small_model()
        inputs = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            model.save(path)
            restored = GPTModel.load(path)

        np.testing.assert_array_equal(model.forward(inputs).data, restored.forward(inputs).data)
        self.assertEqual(model.parameter_count, restored.parameter_count)

    def test_adamw_updates_parameters_and_clips_gradients(self) -> None:
        model = self.small_model()
        inputs = np.array([[0, 1, 2, 3]], dtype=np.int64)
        targets = np.array([[1, 2, 3, 4]], dtype=np.int64)
        optimizer = AdamW(model.parameters(), learning_rate=1e-2, gradient_clip_norm=0.1)
        before = [parameter.data.copy() for parameter in model.parameters()]
        loss = model.loss(inputs, targets)
        loss.backward()
        stats = optimizer.step()

        self.assertGreater(stats["gradient_norm"], 0.0)
        self.assertLess(stats["clip_scale"], 1.0)
        self.assertTrue(any(not np.array_equal(before[i], parameter.data) for i, parameter in enumerate(model.parameters())))

    def test_fixed_batch_can_be_overfit(self) -> None:
        model = self.small_model(seed=9)
        inputs = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
        targets = np.array([[1, 2, 3, 4], [2, 3, 4, 0]], dtype=np.int64)
        optimizer = AdamW(
            model.parameters(),
            learning_rate=5e-2,
            weight_decay=0.0,
            gradient_clip_norm=10.0,
        )
        initial_loss = float(model.loss(inputs, targets).data)
        for _ in range(80):
            model.zero_grad()
            loss = model.loss(inputs, targets)
            loss.backward()
            optimizer.step()
        final_loss = float(model.loss(inputs, targets).data)

        self.assertLess(final_loss, initial_loss * 0.2)
        self.assertLess(final_loss, 0.5)

    def test_evaluation_batches_score_each_transition_once(self) -> None:
        tokens = np.arange(10, dtype=np.int64)
        batches = list(iter_evaluation_batches(tokens, batch_size=2, sequence_length=4))
        scored = sum(int(targets.size) for _, targets in batches)

        self.assertEqual(scored, len(tokens) - 1)
        np.testing.assert_array_equal(batches[0][0], [[0, 1, 2, 3], [4, 5, 6, 7]])
        np.testing.assert_array_equal(batches[-1][0], [[8]])

    def test_sampling_is_seeded_and_keeps_prompt(self) -> None:
        model = self.small_model()
        tokenizer = CharacterTokenizer.from_text("abcde")
        generated_one = sample(model, tokenizer, "abc", max_new_tokens=8, seed=11, temperature=1.0)
        generated_two = sample(model, tokenizer, "abc", max_new_tokens=8, seed=11, temperature=1.0)

        self.assertEqual(generated_one, generated_two)
        self.assertTrue(generated_one.startswith("abc"))
        self.assertEqual(len(generated_one), 11)


if __name__ == "__main__":
    unittest.main()
