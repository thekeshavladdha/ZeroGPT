import unittest

from data.prepare import clean_corpus, split_document


class CleanCorpusTests(unittest.TestCase):
    def test_extracts_story_and_removes_wrapper_and_illustration(self) -> None:
        source = """Gutenberg header and license
ADVENTURES OF SHERLOCK HOLMES

Adventure I

A SCANDAL IN BOHEMIA

The first wrapped
line of the story.

[Illustration:
caption text
]

The final paragraph.
THE END
Publisher advertisements
"""

        cleaned = clean_corpus(source)

        self.assertTrue(cleaned.startswith("ADVENTURES OF SHERLOCK HOLMES\n\nAdventure I"))
        self.assertIn("The first wrapped line of the story.", cleaned)
        self.assertIn("A SCANDAL IN BOHEMIA", cleaned)
        self.assertNotIn("Gutenberg", cleaned)
        self.assertNotIn("Illustration", cleaned)
        self.assertNotIn("Publisher advertisements", cleaned)

    def test_requires_story_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            clean_corpus("not a Project Gutenberg story")

    def test_rejects_unclosed_illustration_caption(self) -> None:
        source = """ADVENTURES OF SHERLOCK HOLMES

Adventure I

[Illustration: missing closing bracket
THE END
"""

        with self.assertRaises(ValueError):
            clean_corpus(source)


class SplitDocumentTests(unittest.TestCase):
    def test_keeps_document_order_and_splits_on_paragraph_boundary(self) -> None:
        text = "one\n\ntwo\n\nthree\n\nfour"

        train, validation = split_document(text, train_fraction=0.5)

        self.assertEqual(train + validation, text)
        self.assertTrue(train.endswith("\n\n"))
        self.assertTrue(validation.startswith("three"))

    def test_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            split_document("one\n\ntwo", train_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
