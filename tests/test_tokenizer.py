import unittest

import numpy as np

from zerogpt.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_uses_stable_sorted_vocabulary_and_round_trips(self) -> None:
        tokenizer = CharacterTokenizer.from_text("cab a")

        self.assertEqual(tokenizer.itos, (" ", "a", "b", "c"))
        encoded = tokenizer.encode("cab a")
        np.testing.assert_array_equal(encoded, np.array([3, 1, 2, 0, 1]))
        self.assertEqual(tokenizer.decode(encoded), "cab a")

    def test_combines_split_vocabulary(self) -> None:
        tokenizer = CharacterTokenizer.from_texts(("abc", "cde"))

        self.assertEqual(tokenizer.vocab_size, 5)
        self.assertEqual(tokenizer.decode(tokenizer.encode("edca")), "edca")

    def test_rejects_unknown_or_invalid_tokens(self) -> None:
        tokenizer = CharacterTokenizer.from_text("ab")

        with self.assertRaises(ValueError):
            tokenizer.encode("abc")
        with self.assertRaises(TypeError):
            tokenizer.decode(np.array([0.0]))
        with self.assertRaises(IndexError):
            tokenizer.decode(np.array([2]))


if __name__ == "__main__":
    unittest.main()
