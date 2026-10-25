import unittest

from data.prepare import clean_book, split_document


class CleanBookTests(unittest.TestCase):
    def test_extracts_between_markers_and_reflows_paragraphs(self) -> None:
        source = (
            "The Project Gutenberg header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TOM SAWYER ***\n"
            "\n"
            "She said\n"
            "hello.\n"
            "\n"
            "[Illustration: a caption]\n"
            "\n"
            "THE END.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TOM SAWYER ***\n"
            "License text\n"
        )

        cleaned = clean_book(source, "Tom Sawyer")

        self.assertEqual(cleaned, "She said hello.\n\nTHE END.")
        self.assertNotIn("Gutenberg", cleaned)
        self.assertNotIn("Illustration", cleaned)
        self.assertNotIn("License", cleaned)

    def test_requires_gutenberg_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            clean_book("not a Project Gutenberg book", "Untitled")

    def test_rejects_end_marker_before_start_marker(self) -> None:
        source = (
            "*** END OF THE PROJECT GUTENBERG EBOOK A ***\n"
            "body\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK A ***\n"
        )
        with self.assertRaises(ValueError):
            clean_book(source, "A")


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
