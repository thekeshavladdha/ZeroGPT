"""Deterministic token batching utilities shared by baseline and GPT runs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _validate_tokens(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("Tokens must be a one-dimensional array")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError("Tokens must contain integer ids")
    return tokens.astype(np.int64, copy=False)


def iter_token_pairs(
    tokens: np.ndarray,
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(context, target)`` pairs exactly once in deterministic batches."""
    tokens = _validate_tokens(tokens)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(tokens) < 2:
        raise ValueError("At least two tokens are required to form pairs")

    contexts = tokens[:-1]
    targets = tokens[1:]
    order = np.arange(len(contexts), dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        yield contexts[indices], targets[indices]


def iter_sequence_batches(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    *,
    seed: int = 0,
    shuffle: bool = False,
    drop_last: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield fixed-length next-token windows for decoder-only training."""
    tokens = _validate_tokens(tokens)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    window_count = len(tokens) - sequence_length
    if window_count <= 0:
        raise ValueError("The token sequence is too short for the requested context length")

    starts = np.arange(window_count, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(starts)

    for start in range(0, len(starts), batch_size):
        batch_starts = starts[start : start + batch_size]
        if drop_last and len(batch_starts) < batch_size:
            break
        inputs = np.stack([tokens[index : index + sequence_length] for index in batch_starts])
        targets = np.stack([tokens[index + 1 : index + sequence_length + 1] for index in batch_starts])
        yield inputs, targets
