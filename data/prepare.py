"""Prepare public-domain Project Gutenberg books for language-model training."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

# The V2 corpus: the Sherlock Holmes canon plus two Mark Twain novels.
BOOKS: tuple[tuple[int, str], ...] = (
    (1661, "The Adventures of Sherlock Holmes"),
    (834, "The Memoirs of Sherlock Holmes"),
    (2852, "The Hound of the Baskervilles"),
    (108, "The Return of Sherlock Holmes"),
    (2343, "The Adventure of Wisteria Lodge"),
    (74, "The Adventures of Tom Sawyer"),
    (76, "Adventures of Huckleberry Finn"),
)

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
START_MARKER = re.compile(r"(?im)^\*{3}[ \t]*START OF THE PROJECT GUTENBERG EBOOK.*$")
END_MARKER = re.compile(r"(?im)^\*{3}[ \t]*END OF THE PROJECT GUTENBERG EBOOK.*$")
ILLUSTRATION = re.compile(r"(?ms)^[ \t]*\[Illustration:.*?\][ \t]*(?:\n|$)")


def fetch_book(book_id: int, raw_dir: Path = RAW_DIR) -> Path:
    """Download a Project Gutenberg ebook once and return the cached file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pg{book_id}.txt"
    if not path.exists():
        url = GUTENBERG_URL.format(book_id=book_id)
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


def _join_paragraphs(body: str) -> str:
    """Strip print indentation and join line-wrapped text within paragraphs."""
    paragraphs: list[str] = []
    lines_in_paragraph: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            lines_in_paragraph.append(line)
        elif lines_in_paragraph:
            paragraphs.append(" ".join(lines_in_paragraph))
            lines_in_paragraph = []
    if lines_in_paragraph:
        paragraphs.append(" ".join(lines_in_paragraph))
    if not paragraphs:
        raise ValueError("The extracted story text is empty")
    return "\n\n".join(paragraphs)


def clean_book(raw_text: str, title: str) -> str:
    """Extract one book's text between the Gutenberg markers and reflow paragraphs."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError(f"Could not locate Gutenberg boundaries in {title!r}")
    body = text[start.end() : end.start()]
    illustration_count = body.count("[Illustration:")
    illustrations = list(ILLUSTRATION.finditer(body))
    if len(illustrations) != illustration_count:
        raise ValueError(f"Found an illustration caption without a matching closing bracket in {title!r}")
    return _join_paragraphs(ILLUSTRATION.sub("", body))


def split_document(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    """Split in document order at the paragraph boundary nearest the target."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    target = round(len(text) * train_fraction)
    boundaries = [match.end() for match in re.finditer(r"\n\n", text)]
    boundaries = [boundary for boundary in boundaries if 0 < boundary < len(text)]
    if not boundaries:
        raise ValueError("The cleaned text needs at least two paragraphs to split")

    split_at = min(boundaries, key=lambda boundary: abs(boundary - target))
    train_text, validation_text = text[:split_at], text[split_at:]
    if not train_text.strip() or not validation_text.strip():
        raise ValueError("The selected split produced an empty training or validation set")
    return train_text, validation_text


def prepare(output_dir: Path, train_fraction: float = 0.9) -> dict[str, object]:
    sources: list[dict[str, object]] = []
    cleaned_parts: list[str] = []
    for book_id, title in BOOKS:
        raw_text = fetch_book(book_id).read_text(encoding="utf-8-sig")
        cleaned = clean_book(raw_text, title)
        cleaned_parts.append(cleaned)
        sources.append({"gutenberg_id": book_id, "title": title, "characters": len(cleaned)})

    cleaned_text = "\n\n".join(cleaned_parts)
    train_text, validation_text = split_document(cleaned_text, train_fraction)

    # The character tokenizer is fitted on the training split only, so every
    # validation character must already appear in the training text.
    missing = sorted(set(validation_text).difference(train_text))
    if missing:
        raise ValueError(f"Validation characters absent from the training text: {missing!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (output_dir / "validation.txt").write_text(validation_text, encoding="utf-8")

    summary: dict[str, object] = {
        "sources": sources,
        "cleaned_characters": len(cleaned_text),
        "vocabulary_size": len(set(cleaned_text)),
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "train_fraction": round(len(train_text) / len(cleaned_text), 6),
        "split_method": "document order, nearest paragraph boundary to requested fraction",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    args = parser.parse_args()

    summary = prepare(args.output_dir, args.train_fraction)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
