"""Prepare the supplied Sherlock Holmes text for character-level training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "dataset.txt"
DEFAULT_OUTPUT = ROOT / "data" / "processed"

BOOK_START = re.compile(
    r"(?m)^ADVENTURES OF SHERLOCK HOLMES[ \t]*\n(?:[ \t]*\n)+[ \t]*Adventure I[ \t]*$"
)
BOOK_END = re.compile(r"(?m)^THE END[ \t]*$")
ILLUSTRATION = re.compile(r"(?ms)^[ \t]*\[Illustration:.*?\][ \t]*(?:\n|$)")


def clean_corpus(raw_text: str) -> str:
    """Extract the stories and normalize print layout without changing words."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    starts = list(BOOK_START.finditer(text))
    if len(starts) != 1:
        raise ValueError(f"Expected one book/story start marker; found {len(starts)}")

    end = BOOK_END.search(text, starts[0].end())
    if end is None:
        raise ValueError("Could not find the final 'THE END' story marker")

    body = text[starts[0].start() : end.end()]
    illustration_count = body.count("[Illustration:")
    illustrations = list(ILLUSTRATION.finditer(body))
    if len(illustrations) != illustration_count:
        raise ValueError("Found an illustration caption without a matching closing bracket")
    body = ILLUSTRATION.sub("", body)

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


def prepare(input_path: Path, output_dir: Path, train_fraction: float = 0.9) -> dict[str, object]:
    raw_text = input_path.read_text(encoding="utf-8-sig")
    cleaned = clean_corpus(raw_text)
    train_text, validation_text = split_document(cleaned, train_fraction)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (output_dir / "validation.txt").write_text(validation_text, encoding="utf-8")

    total_chars = len(cleaned)
    train_chars = len(train_text)
    summary: dict[str, object] = {
        "source_file": input_path.name,
        "cleaned_characters": total_chars,
        "vocabulary_size": len(set(cleaned)),
        "train_characters": train_chars,
        "validation_characters": len(validation_text),
        "train_fraction": round(train_chars / total_chars, 6),
        "split_method": "document order, nearest paragraph boundary to requested fraction",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    args = parser.parse_args()

    summary = prepare(args.input, args.output_dir, args.train_fraction)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
