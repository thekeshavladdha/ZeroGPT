"""Train, evaluate, checkpoint, and sample from the tiny NumPy GPT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerogpt.batching import iter_sequence_batches
from zerogpt.bigram import write_json
from zerogpt.gpt import GPTConfig, GPTModel
from zerogpt.optim import AdamW
from zerogpt.system import peak_memory_mb
from zerogpt.tokenizer import BytePairTokenizer, CharacterTokenizer


DEFAULT_TRAIN = ROOT / "data" / "processed" / "train.txt"
DEFAULT_VALIDATION = ROOT / "data" / "processed" / "validation.txt"
DEFAULT_OUTPUT = ROOT / "runs" / "gpt"
DEFAULT_PROMPT = "Sherlock Holmes"


def report_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        relative = path.resolve()
    return relative.as_posix()


def iter_evaluation_batches(
    tokens: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
):
    """Yield non-overlapping sequential blocks so every transition is scored once."""
    if tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError("Evaluation requires at least two one-dimensional tokens")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")

    input_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for start in range(0, len(tokens) - 1, sequence_length):
        end = min(start + sequence_length, len(tokens) - 1)
        inputs = tokens[start:end]
        targets = tokens[start + 1 : end + 1]
        if len(inputs) < sequence_length:
            if input_blocks:
                yield np.stack(input_blocks), np.stack(target_blocks)
            yield inputs[None, :], targets[None, :]
            return
        input_blocks.append(inputs)
        target_blocks.append(targets)
        if len(input_blocks) == batch_size:
            yield np.stack(input_blocks), np.stack(target_blocks)
            input_blocks = []
            target_blocks = []
    if input_blocks:
        yield np.stack(input_blocks), np.stack(target_blocks)


def evaluate(
    model: GPTModel,
    tokens: np.ndarray,
    *,
    batch_size: int,
    seed: int = 0,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    del seed  # Evaluation is sequential; the argument keeps the call contract explicit.
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for inputs, targets in iter_evaluation_batches(
        tokens,
        batch_size=batch_size,
        sequence_length=model.config.context_length,
    ):
        if max_batches is not None and batches >= max_batches:
            break
        loss = model.loss(inputs, targets)
        token_count = int(targets.size)
        total_loss += float(loss.data) * token_count
        total_tokens += token_count
        batches += 1
    if total_tokens == 0:
        raise ValueError("Evaluation produced no tokens")
    return {
        "loss_nats": total_loss / total_tokens,
        "tokens": total_tokens,
        "batches": batches,
    }


def sample(
    model: GPTModel,
    tokenizer: CharacterTokenizer | BytePairTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    temperature: float,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    generated = tokenizer.encode(prompt).tolist()
    if not generated:
        raise ValueError("The sampling prompt cannot be empty")
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        context = np.asarray(generated[-model.config.context_length :], dtype=np.int64)[None, :]
        logits = model.forward(context).data[0, -1].astype(np.float64)
        logits = logits / temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        generated.append(int(rng.choice(model.config.vocab_size, p=probabilities)))
    return tokenizer.decode(generated)


def build_tokenizer(
    train_text: str,
    validation_text: str,
    *,
    tokenizer_type: str,
    vocab_limit: int,
) -> CharacterTokenizer | BytePairTokenizer:
    """Construct a tokenizer from the training and validation texts."""
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
    seed: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    eval_interval: int,
    warmup_steps: int,
    prompt: str,
    sample_tokens: int,
    temperature: float,
    tokenizer_type: str,
    vocab_limit: int,
) -> dict[str, object]:
    if steps <= 0 or batch_size <= 0 or eval_batch_size <= 0 or eval_interval <= 0:
        raise ValueError("steps, batch sizes, and eval_interval must be positive")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    train_text = train_path.read_text(encoding="utf-8")
    validation_text = validation_path.read_text(encoding="utf-8")
    tokenizer = build_tokenizer(train_text, validation_text, tokenizer_type=tokenizer_type, vocab_limit=vocab_limit)
    train_tokens = tokenizer.encode(train_text)
    validation_tokens = tokenizer.encode(validation_text)
    prompt_tokens = tokenizer.encode(prompt)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        num_layers=2,
        num_heads=8,
        model_width=64,
        mlp_width=256,
    )
    model = GPTModel(config, seed=seed)
    optimizer = AdamW(
        model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    started = time.perf_counter()
    tokens_seen = 0
    step = 0
    epoch = 0
    batch_iterator = iter_sequence_batches(
        train_tokens,
        batch_size=batch_size,
        sequence_length=config.context_length,
        seed=seed + epoch,
        shuffle=True,
        drop_last=False,
    )

    def learning_rate_for(current_step: int) -> float:
        """Linear warm-up followed by cosine decay to 10% of the base rate."""
        if current_step < warmup_steps:
            return learning_rate * (current_step + 1) / warmup_steps
        progress = (current_step - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return learning_rate * (0.1 + 0.9 * cosine)

    with log_path.open("w", encoding="utf-8") as log_file:
        while step < steps:
            try:
                inputs, targets = next(batch_iterator)
            except StopIteration:
                epoch += 1
                batch_iterator = iter_sequence_batches(
                    train_tokens,
                    batch_size=batch_size,
                    sequence_length=config.context_length,
                    seed=seed + epoch,
                    shuffle=True,
                    drop_last=False,
                )
                continue
            step += 1
            model.zero_grad()
            loss = model.loss(inputs, targets)
            loss_value = float(loss.data)
            loss.backward()
            rate = learning_rate_for(step)
            optimizer_stats = optimizer.step(learning_rate=rate)
            tokens_seen += int(targets.size)
            elapsed = time.perf_counter() - started
            record = {
                "event": "train_step",
                "step": step,
                "epoch": epoch,
                "loss_nats": loss_value,
                "learning_rate": rate,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / elapsed,
                **optimizer_stats,
            }
            if step % eval_interval == 0 or step == steps:
                train_eval = evaluate(
                    model,
                    train_tokens,
                    batch_size=eval_batch_size,
                    seed=seed,
                    max_batches=16,
                )
                validation = evaluate(
                    model,
                    validation_tokens,
                    batch_size=eval_batch_size,
                    seed=seed,
                    max_batches=16,
                )
                record["train_loss_sample_nats"] = train_eval["loss_nats"]
                record["validation_loss_sample_nats"] = validation["loss_nats"]
                record["overfit_gap_nats"] = (
                    train_eval["loss_nats"] - validation["loss_nats"]
                )
            log_file.write(json.dumps(record) + "\n")

    training_elapsed_seconds = time.perf_counter() - started
    full_validation = evaluate(
        model,
        validation_tokens,
        batch_size=eval_batch_size,
        seed=seed,
    )
    full_training = evaluate(
        model,
        train_tokens,
        batch_size=eval_batch_size,
        seed=seed,
        max_batches=64,
    )
    checkpoint_path = output_dir / "checkpoint.npz"
    model.save(checkpoint_path)
    loaded_model = GPTModel.load(checkpoint_path)
    check_inputs, _ = next(
        iter_sequence_batches(
            validation_tokens,
            batch_size=min(batch_size, 2),
            sequence_length=config.context_length,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
    )
    original_logits = model.forward(check_inputs).data
    restored_logits = loaded_model.forward(check_inputs).data
    checkpoint_max_logit_difference = float(np.max(np.abs(original_logits - restored_logits)))
    generated = sample(
        loaded_model,
        tokenizer,
        prompt,
        max_new_tokens=sample_tokens,
        seed=seed,
        temperature=temperature,
    )
    (output_dir / "sample.txt").write_text(generated, encoding="utf-8")
    write_json(output_dir / "tokenizer.json", tokenizer.to_dict())
    config_artifact = {
        "train_path": report_path(train_path),
        "validation_path": report_path(validation_path),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "eval_interval": eval_interval,
        "prompt": prompt,
        "sample_tokens": sample_tokens,
        "temperature": temperature,
        "model": config.to_dict(),
        "parameter_count": model.parameter_count,
    }
    write_json(output_dir / "config.json", config_artifact)
    bigram_results_path = ROOT / "runs" / "bigram" / "results.json"
    bigram_results = json.loads(bigram_results_path.read_text(encoding="utf-8"))
    validation_loss = float(full_validation["loss_nats"])
    bigram_loss = float(bigram_results["validation_loss_nats"])
    elapsed_seconds = time.perf_counter() - started
    results = {
        "model": "tiny_decoder_gpt",
        "tokenizer_type": tokenizer_type,
        "vocab_limit": vocab_limit,
        "seed": seed,
        "vocabulary_size": tokenizer.vocab_size,
        "train_characters": len(train_text),
        "validation_characters": len(validation_text),
        "parameter_count": model.parameter_count,
        "steps": steps,
        "tokens_seen": tokens_seen,
        "train_loss_nats_sample": full_training["loss_nats"],
        "validation_loss_nats": validation_loss,
        "validation_tokens": full_validation["tokens"],
        "bigram_validation_loss_nats": bigram_loss,
        "validation_improvement_nats": bigram_loss - validation_loss,
        "beats_bigram": validation_loss < bigram_loss,
        "elapsed_seconds": elapsed_seconds,
        "training_elapsed_seconds": training_elapsed_seconds,
        "train_tokens_per_second": tokens_seen / training_elapsed_seconds,
        "end_to_end_tokens_per_second": tokens_seen / elapsed_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "checkpoint_max_logit_difference": checkpoint_max_logit_difference,
        "checkpoint": report_path(checkpoint_path),
        "config": report_path(output_dir / "config.json"),
        "log": report_path(log_path),
        "sample": report_path(output_dir / "sample.txt"),
    }
    write_json(output_dir / "results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--tokenizer-type",
        choices=("character", "byte_pair"),
        default="byte_pair",
        help="Subword BPE is the default for new runs; character is available for comparison.",
    )
    parser.add_argument("--vocab-limit", type=int, default=512)
    args = parser.parse_args()
    results = run(
        args.train,
        args.validation,
        args.output_dir,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        eval_interval=args.eval_interval,
        warmup_steps=args.warmup_steps,
        prompt=args.prompt,
        sample_tokens=args.sample_tokens,
        temperature=args.temperature,
        tokenizer_type=args.tokenizer_type,
        vocab_limit=args.vocab_limit,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
