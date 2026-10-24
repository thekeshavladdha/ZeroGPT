# Dataset preparation

The input is the supplied `dataset.txt`. Its header identifies the source as
Project Gutenberg eBook **Adventures of Sherlock Holmes**, ebook 48320, by
Arthur Conan Doyle. The source page is <https://www.gutenberg.org/ebooks/48320>.
The supplied file includes Project Gutenberg's license text and source notice;
retain the original file and follow those terms when sharing it. Check the rules
that apply where you live before redistributing the source or derived text.

Run `python data/prepare.py` from the repository root. The script extracts the
book title and twelve stories through the final `THE END`, removes illustration
captions, strips print indentation, joins line-wrapped text within paragraphs,
and preserves paragraph and story-heading boundaries. This excludes the
contents pages, publisher advertisements, transcriber notes, and Gutenberg
license boilerplate from model input. It does not modify `dataset.txt`.

The cleaned text is split in document order at the paragraph boundary nearest
90% of its character count. The script writes `train.txt`, `validation.txt`,
and `summary.json` under `data/processed/`. The summary reports the cleaned
character count, unique-character vocabulary size, split sizes, and actual
training fraction.
