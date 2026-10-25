import unittest

import numpy as np

from zerogpt.tokenizer import BytePairTokenizer, CharacterTokenizer, load_tokenizer


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


class BytePairTokenizerTests(unittest.TestCase):
    def test_round_trips_and_compresses_repeated_text(self) -> None:
        text = "Sherlock Holmes\nSherlock Holmes said hello world\nhello world hello"
        tokenizer = BytePairTokenizer.from_texts([text], vocab_limit=300)
        encoded = tokenizer.encode(text)
        self.assertLess(len(encoded), len(text))
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_is_deterministic(self) -> None:
        texts = ["aaabbb", "abc abc", "the cat sat"]
        one = BytePairTokenizer.from_texts(texts, vocab_limit=256 + 32)
        two = BytePairTokenizer.from_texts(texts, vocab_limit=256 + 32)
        self.assertEqual(one.merges, two.merges)
        self.assertEqual(one.encode("aaabbb").tolist(), two.encode("aaabbb").tolist())

    def test_handles_unicode_without_unknown_tokens(self) -> None:
        tokenizer = BytePairTokenizer.from_texts(
            ["Héllo Wörld", "emoji 🧪 test", "UPPER lower Mixed123\n"],
            vocab_limit=300,
        )
        for sample in ["Héllo Wörld", "emoji 🧪 test", "UPPER lower Mixed123\n"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_save_and_load_preserves_merges_and_encoding(self) -> None:
        tokenizer = BytePairTokenizer.from_texts(["the cat sat on the mat"], vocab_limit=300)
        restored = BytePairTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.merges, tokenizer.merges)
        np.testing.assert_array_equal(
            restored.encode("the cat sat"), tokenizer.encode("the cat sat")
        )

    def test_rejects_invalid_vocab_limit_and_merge_count(self) -> None:
        # An empty merge list with a 256-token limit is a valid byte-level tokenizer.
        bare = BytePairTokenizer([], vocab_limit=256)
        self.assertEqual(bare.vocab_size, 256)
        # vocab_limit below the 256 byte vocabulary is invalid.
        with self.assertRaises(ValueError):
            BytePairTokenizer([], vocab_limit=128)
        # Merges cannot exceed the room left above the byte vocabulary.
        with self.assertRaises(ValueError):
            BytePairTokenizer([(0, 1), (1, 2)], vocab_limit=257)
        with self.assertRaises(ValueError):
            BytePairTokenizer.from_dict({"type": "byte_pair"})


class LoadTokenizerTests(unittest.TestCase):
    def test_dispatches_on_type_field(self) -> None:
        character = CharacterTokenizer.from_text("abc")
        byte_pair = BytePairTokenizer.from_texts(["abc abc"], vocab_limit=300)
        self.assertIsInstance(load_tokenizer(character.to_dict()), CharacterTokenizer)
        self.assertIsInstance(load_tokenizer(byte_pair.to_dict()), BytePairTokenizer)

    def test_character_state_without_type_field_still_loads(self) -> None:
        state = CharacterTokenizer.from_text("abc").to_dict()
        self.assertIsInstance(load_tokenizer(state), CharacterTokenizer)

    def test_rejects_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            load_tokenizer({"type": "magic"})


if __name__ == "__main__":
    unittest.main()
