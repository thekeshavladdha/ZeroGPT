"""Train and measure the character-level bigram baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerogpt.bigram import BigramModel, write_json
from zerogpt.system import peak_memory_mb
from zerogpt.tokenizer import BytePairTokenizer, CharacterTokenizer


DEFAULT_TRAIN = ROOT / "data" / "processed" / "train.txt"
DEFAULT_VALIDATION = ROOT / "data" / "processed" / "validation.txt"
DEFAULT_OUTPUT = ROOT / "runs" / "bigram"


def report_path(path: Path) -> str:
    """Prefer repository-relative paths so saved configs remain portable."""
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = path.resolve()
    return relative.as_posix()


def build_tokenizer(
    train_text: str,
    validation_text: str,
    *,
    tokenizer_type: str,
    vocab_limit: int,
) -> CharacterTokenizer | BytePairTokenizer:
    if tokenizer_type == "character":
        return CharacterTokenizer.from_texts((train_text, validation_text))
    if tokenizer_type == "byte_pair":
        return BytePairTokenizer.from_texts((train_text, validation_text), vocab_limit=vocab_limit)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")


def run(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    seed: int,
    smoothing: float,
    shuffle: bool,
    tokenizer_type: str,
    vocab_limit: int,
) -> dict[str, object]:
    train_text = train_path.read_text(encoding="utf-8")
    validation_text = validation_path.read_text(encoding="utf-8")
    tokenizer = build_tokenizer(
        train_text,
        validation_text,
        tokenizer_type=tokenizer_type,
        vocab_limit=vocab_limit,
    )
    train_tokens = tokenizer.encode(train_text)
    validation_tokens = tokenizer.encode(validation_text)

    model = BigramModel(tokenizer.vocab_size, smoothing=smoothing)
    log_path = output_dir / "train.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        def log_batch(step: int, processed_tokens: int) -> None:
            log_file.write(
                json.dumps(
                    {
                        "event": "train_batch",
                        "step": step,
                        "processed_tokens": processed_tokens,
                    }
                )
                + "\n"
            )

        processed_tokens = model.fit(
            train_tokens,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            on_batch=log_batch,
        )
        elapsed_seconds = time.perf_counter() - started
        validation = model.evaluate(validation_tokens)
        training = model.evaluate(train_tokens)
        log_file.write(
            json.dumps(
                {
                    "event": "evaluation",
                    "train_loss_nats": training["loss_nats"],
                    "validation_loss_nats": validation["loss_nats"],
                }
            )
            + "\n"
        )

    checkpoint_path = output_dir / "checkpoint.npz"
    model.save(checkpoint_path)
    write_json(output_dir / "tokenizer.json", tokenizer.to_dict())
    config = {
        "train_path": report_path(train_path),
        "validation_path": report_path(validation_path),
        "batch_size": batch_size,
        "seed": seed,
        "smoothing": smoothing,
        "shuffle": shuffle,
        "model": model.to_dict(),
    }
    write_json(output_dir / "config.json", config)
    results = {
        "model": "character_bigram",
        "tokenizer_type": tokenizer_type,
        "seed": seed,
        "vocabulary_size": tokenizer.vocab_size,
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "processed_tokens": processed_tokens,
        "train_loss_nats": training["loss_nats"],
        "validation_loss_nats": validation["loss_nats"],
        "elapsed_seconds": elapsed_seconds,
        "train_tokens_per_second": processed_tokens / elapsed_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "checkpoint": report_path(checkpoint_path),
        "config": report_path(output_dir / "config.json"),
        "log": report_path(log_path),
    }
    write_json(output_dir / "results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--tokenizer-type",
        choices=("character", "byte_pair"),
        default="byte_pair",
    )
    parser.add_argument("--vocab-limit", type=int, default=512)
    args = parser.parse_args()
    results = run(
        args.train,
        args.validation,
        args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        smoothing=args.smoothing,
        shuffle=args.shuffle,
        tokenizer_type=args.tokenizer_type,
        vocab_limit=args.vocab_limit,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
