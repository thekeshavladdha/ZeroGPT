import unittest

import numpy as np

from zerogpt.batching import iter_sequence_batches, iter_token_pairs


class BatchingTests(unittest.TestCase):
    def test_pair_batches_cover_each_transition_in_order(self) -> None:
        tokens = np.arange(6, dtype=np.int64)
        batches = list(iter_token_pairs(tokens, batch_size=2))

        np.testing.assert_array_equal(batches[0][0], [0, 1])
        np.testing.assert_array_equal(batches[0][1], [1, 2])
        np.testing.assert_array_equal(batches[-1][0], [4])
        np.testing.assert_array_equal(batches[-1][1], [5])

    def test_shuffled_batches_are_reproducible(self) -> None:
        tokens = np.arange(9, dtype=np.int64)
        first = list(iter_token_pairs(tokens, batch_size=3, seed=7, shuffle=True))
        second = list(iter_token_pairs(tokens, batch_size=3, seed=7, shuffle=True))

        for (first_x, first_y), (second_x, second_y) in zip(first, second):
            np.testing.assert_array_equal(first_x, second_x)
            np.testing.assert_array_equal(first_y, second_y)

    def test_sequence_batches_shift_targets_by_one(self) -> None:
        tokens = np.arange(7, dtype=np.int64)
        inputs, targets = next(iter_sequence_batches(tokens, batch_size=2, sequence_length=3))

        np.testing.assert_array_equal(inputs, [[0, 1, 2], [1, 2, 3]])
        np.testing.assert_array_equal(targets, [[1, 2, 3], [2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
