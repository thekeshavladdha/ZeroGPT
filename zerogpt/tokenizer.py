"""Tokenizers for the ZeroGPT corpus: character-level and byte-pair subword."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class CharacterTokenizer:
    """Map the characters in a corpus to stable integer token ids."""

    def __init__(self, characters: Iterable[str]) -> None:
        vocabulary = tuple(characters)
        if not vocabulary:
            raise ValueError("The tokenizer vocabulary cannot be empty")
        if any(not isinstance(character, str) or len(character) != 1 for character in vocabulary):
            raise ValueError("Each vocabulary entry must be one character")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("The tokenizer vocabulary cannot contain duplicates")

        self.itos = vocabulary
        self.stoi = {character: index for index, character in enumerate(vocabulary)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Build a tokenizer from a single corpus in sorted character order."""
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharacterTokenizer":
        """Build one vocabulary from multiple text splits."""
        characters: set[str] = set()
        for text in texts:
            characters.update(text)
        return cls(sorted(characters))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        """Return token ids for ``text`` as a one-dimensional integer array."""
        unknown = sorted(set(text).difference(self.stoi))
        if unknown:
            raise ValueError(f"Text contains characters outside the vocabulary: {unknown!r}")
        return np.fromiter((self.stoi[character] for character in text), dtype=np.int64, count=len(text))

    def decode(self, tokens: Iterable[int] | np.ndarray) -> str:
        """Decode integer token ids back to their original characters."""
        token_array = np.asarray(tokens)
        if not np.issubdtype(token_array.dtype, np.integer):
            raise TypeError("Token ids must be integers")
        if token_array.size and (token_array.min() < 0 or token_array.max() >= self.vocab_size):
            raise IndexError("Token id is outside the vocabulary")
        return "".join(self.itos[int(token)] for token in token_array.reshape(-1))

    def to_dict(self) -> dict[str, list[str]]:
        """Return a JSON-serializable tokenizer description."""
        return {"characters": list(self.itos)}

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "CharacterTokenizer":
        characters = state.get("characters")
        if not isinstance(characters, list):
            raise ValueError("Tokenizer state must contain a list of characters")
        return cls(characters)


class BytePairTokenizer:
    """Byte-level BPE tokenizer with a learned, deterministic merge table.

    The vocabulary starts as the 256 byte values. Training repeatedly merges
    the most frequent adjacent pair of token ids into a new id, up to
    ``vocab_limit`` total tokens. Encoding applies the learned merges in rank
    order (lowest rank first), which is equivalent to greedily merging the
    highest-priority pair at every position. Because merges are learned from
    bytes rather than Unicode code points, every input string round-trips with
    no unknown-token handling.
    """

    def __init__(self, merges: list[tuple[int, int]], vocab_limit: int = 256) -> None:
        if vocab_limit < 256:
            raise ValueError("vocab_limit must be at least 256 (the byte vocabulary)")
        self.merges: list[tuple[int, int]] = [(int(left), int(right)) for left, right in merges]
        if len(self.merges) > vocab_limit - 256:
            raise ValueError("vocab_limit is too small for the supplied merges")
        self.vocab_limit = vocab_limit
        self._token_to_bytes: list[bytes] | None = None

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def from_texts(cls, texts: Iterable[str], vocab_limit: int = 512) -> "BytePairTokenizer":
        """Learn one merge table from the union of all texts.

        shortcut: merges are learned from a capped, evenly-spaced sample of the
        input bytes so fitting a multi-megabyte corpus stays fast; revisit if
        merge quality at large vocabularies ever looks weak.
        """
        data = cls._sample_bytes(b"".join(text.encode("utf-8") for text in texts))
        return cls(cls._learn_merges(data, vocab_limit), vocab_limit=vocab_limit)

    MERGE_SAMPLE_BYTES = 1_000_000
    MERGE_SAMPLE_WINDOWS = 8

    @classmethod
    def _sample_bytes(cls, data: bytes) -> bytes:
        """Return up to ``MERGE_SAMPLE_BYTES`` bytes in evenly-spaced windows."""
        if len(data) <= cls.MERGE_SAMPLE_BYTES:
            return data
        windows = cls.MERGE_SAMPLE_WINDOWS
        window_size = cls.MERGE_SAMPLE_BYTES // windows
        starts = [round(index * (len(data) - window_size) / (windows - 1)) for index in range(windows)]
        return b"".join(data[start : start + window_size] for start in starts)

    @staticmethod
    def _merge_pair(tokens: list[int], left: int, right: int, new_id: int) -> list[int]:
        """Replace every adjacent ``[left, right]`` in ``tokens`` with ``new_id``."""
        merged: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
                merged.append(new_id)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    @staticmethod
    def _learn_merges(data: bytes, vocab_limit: int) -> list[tuple[int, int]]:
        if vocab_limit <= 256 or len(data) < 2:
            return []
        tokens = list(data)
        merges: list[tuple[int, int]] = []
        for rank in range(vocab_limit - 256):
            counts: dict[tuple[int, int], int] = {}
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            # Most frequent pair; ties broken by smallest pair for determinism.
            best_pair = min(counts, key=lambda pair: (-counts[pair], pair))
            if counts[best_pair] < 2:
                break  # No pair is worth merging.
            merges.append(best_pair)
            tokens = BytePairTokenizer._merge_pair(tokens, *best_pair, 256 + rank)
        return merges

    def encode(self, text: str) -> np.ndarray:
        """Return subword token ids for ``text`` as a one-dimensional array."""
        tokens = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            if len(tokens) < 2:
                break
            tokens = self._merge_pair(tokens, *pair, 256 + rank)
        return np.fromiter(tokens, dtype=np.int64, count=len(tokens))

    def _build_token_bytes(self) -> list[bytes]:
        table: list[bytes] = [bytes([value]) for value in range(256)]
        for left, right in self.merges:
            table.append(table[left] + table[right])
        return table

    @property
    def token_to_bytes(self) -> list[bytes]:
        if self._token_to_bytes is None:
            self._token_to_bytes = self._build_token_bytes()
        return self._token_to_bytes

    def decode(self, tokens: Iterable[int] | np.ndarray) -> str:
        """Decode token ids back to the original UTF-8 string."""
        table = self.token_to_bytes
        byte_values = bytearray()
        for token in np.asarray(tokens).reshape(-1):
            int_token = int(token)
            if int_token < 0 or int_token >= len(table):
                raise IndexError(f"Token id {int_token} is outside the tokenizer vocabulary")
            byte_values.extend(table[int_token])
        # Model sampling can emit token sequences that do not form valid UTF-8.
        # Use the standard error handler so generation never crashes; the
        # replacement character marks where the byte stream was malformed.
        return bytes(byte_values).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "byte_pair",
            "vocab_limit": self.vocab_limit,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "BytePairTokenizer":
        merges_raw = state.get("merges")
        if not isinstance(merges_raw, list):
            raise ValueError("Tokenizer state must contain a list of merges")
        merges = [(int(left), int(right)) for left, right in merges_raw]
        vocab_limit = int(state.get("vocab_limit", 256 + len(merges)))
        return cls(merges, vocab_limit=vocab_limit)


def load_tokenizer(state: dict[str, object]) -> CharacterTokenizer | BytePairTokenizer:
    """Reconstruct a tokenizer from its saved dictionary.

    Dispatches on the ``type`` field so callers do not need to know which
    tokenizer variant was used. Character checkpoints without a ``type``
    field still load for backward compatibility.
    """
    type_name = state.get("type", "character")
    if type_name == "byte_pair":
        return BytePairTokenizer.from_dict(state)
    if type_name == "character":
        return CharacterTokenizer.from_dict(state)
    raise ValueError(f"Unknown tokenizer type: {type_name!r}")
