import tempfile
import unittest
from pathlib import Path

import numpy as np

from zerogpt.bigram import BigramModel


class BigramModelTests(unittest.TestCase):
    def test_fits_transition_counts_and_evaluates_in_nats(self) -> None:
        model = BigramModel(vocab_size=3, smoothing=1.0)
        tokens = np.array([0, 1, 0, 1, 2], dtype=np.int64)

        processed = model.fit(tokens, batch_size=2)

        self.assertEqual(processed, len(tokens))
        np.testing.assert_array_equal(model.start_counts, [1, 0, 0])
        np.testing.assert_array_equal(model.transition_counts, [[0, 2, 0], [1, 0, 1], [0, 0, 0]])
        result = model.evaluate(tokens)
        self.assertEqual(result["tokens"], 4)
        self.assertGreater(result["loss_nats"], 0.0)

    def test_checkpoint_round_trip_preserves_evaluation(self) -> None:
        model = BigramModel(vocab_size=2, smoothing=0.5)
        tokens = np.array([0, 1, 1, 0, 1], dtype=np.int64)
        model.fit(tokens)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            model.save(path)
            restored = BigramModel.load(path)

        self.assertEqual(restored.to_dict(), model.to_dict())
        self.assertEqual(restored.evaluate(tokens), model.evaluate(tokens))

    def test_rejects_tokens_outside_vocabulary(self) -> None:
        model = BigramModel(vocab_size=2)

        with self.assertRaises(IndexError):
            model.fit(np.array([0, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
