"""Deterministic character-level tokenization for the ZeroGPT corpus."""

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
