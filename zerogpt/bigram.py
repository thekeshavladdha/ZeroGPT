"""Count-based character bigram baseline."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .batching import iter_token_pairs


BatchCallback = Callable[[int, int], None]


class BigramModel:
    """Estimate the next character from smoothed count statistics."""

    def __init__(self, vocab_size: int, smoothing: float = 1.0) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.vocab_size = int(vocab_size)
        self.smoothing = float(smoothing)
        self.start_counts = np.zeros(self.vocab_size, dtype=np.int64)
        self.transition_counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int64)

    def reset(self) -> None:
        self.start_counts.fill(0)
        self.transition_counts.fill(0)

    def fit(
        self,
        tokens: np.ndarray,
        *,
        batch_size: int = 8192,
        seed: int = 0,
        shuffle: bool = False,
        on_batch: BatchCallback | None = None,
    ) -> int:
        """Fit counts in deterministic mini-batches and return processed token count."""
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("tokens must be a one-dimensional integer array")
        if len(tokens) < 2:
            raise ValueError("At least two tokens are required to fit a bigram model")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise IndexError("Token id is outside the model vocabulary")

        self.reset()
        self.start_counts[int(tokens[0])] += 1
        processed = 1
        for step, (contexts, targets) in enumerate(
            iter_token_pairs(tokens, batch_size, seed=seed, shuffle=shuffle), start=1
        ):
            np.add.at(self.transition_counts, (contexts, targets), 1)
            processed += len(targets)
            if on_batch is not None:
                on_batch(step, processed)
        return processed

    def _start_log_probabilities(self) -> np.ndarray:
        counts = self.start_counts.astype(np.float64) + self.smoothing
        return np.log(counts / counts.sum())

    def _transition_log_probabilities(self) -> np.ndarray:
        counts = self.transition_counts.astype(np.float64) + self.smoothing
        return np.log(counts / counts.sum(axis=1, keepdims=True))

    def evaluate(self, tokens: np.ndarray) -> dict[str, float | int]:
        """Return mean next-character cross-entropy in nats for a token sequence."""
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("tokens must be a one-dimensional integer array")
        if len(tokens) < 2:
            raise ValueError("At least two tokens are required to evaluate transitions")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise IndexError("Token id is outside the model vocabulary")

        transition_log_probs = self._transition_log_probabilities()
        losses = -transition_log_probs[tokens[:-1], tokens[1:]]
        loss = float(np.mean(np.asarray(losses, dtype=np.float64)))
        return {"loss_nats": loss, "tokens": int(len(tokens) - 1)}

    def save(self, path: Path) -> None:
        """Save count arrays and model metadata in a portable NumPy archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vocab_size=np.asarray(self.vocab_size, dtype=np.int64),
            smoothing=np.asarray(self.smoothing, dtype=np.float64),
            start_counts=self.start_counts,
            transition_counts=self.transition_counts,
        )

    @classmethod
    def load(cls, path: Path) -> "BigramModel":
        with np.load(path) as archive:
            model = cls(
                int(archive["vocab_size"]),
                float(archive["smoothing"]),
            )
            model.start_counts[...] = archive["start_counts"]
            model.transition_counts[...] = archive["transition_counts"]
        return model

    def to_dict(self) -> dict[str, object]:
        """Return compact metadata useful in run reports."""
        return {
            "vocab_size": self.vocab_size,
            "smoothing": self.smoothing,
            "observed_start_tokens": int(self.start_counts.sum()),
            "observed_transitions": int(self.transition_counts.sum()),
        }


def write_json(path: Path, value: object) -> None:
    """Write an indented UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
