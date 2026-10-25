# Dataset preparation

The V2 corpus is public-domain Project Gutenberg ebooks: the Sherlock Holmes
canon by Arthur Conan Doyle (ebooks 1661, 834, 2852, 108, 2343) plus Mark
Twain's *The Adventures of Tom Sawyer* (74) and *Adventures of Huckleberry
Finn* (76). Each book's page is `https://www.gutenberg.org/ebooks/<id>`.
Project Gutenberg texts carry the Project Gutenberg License; retain those
terms when redistributing the source or derived text.

Run `python data/prepare.py` from the repository root. The script downloads
each ebook once into `data/raw/` (skipping cached files), extracts the text
between the `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers,
removes illustration captions, strips print indentation, and joins
line-wrapped text within paragraphs. V1's single-book corpus (`dataset.txt`,
ebook 48320) remains in git history.

The cleaned books are concatenated in the order listed above and split in
document order at the paragraph boundary nearest 90% of the character count,
so the validation split is the corpus tail. The script fails if any
validation character is absent from the training text, because the character
tokenizer is fitted on the training split only. Outputs under
`data/processed/`: `train.txt`, `validation.txt`, and `summary.json` (per-book
character counts, total, vocabulary size, split sizes).
