"""Tiny decoder-only GPT built on the project's NumPy autograd engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .autograd import Tensor, causal_attention, cross_entropy, embedding, gelu, layer_norm, linear


@dataclass(frozen=True)
class GPTConfig:
    """Configuration for the small decoder-only transformer."""

    vocab_size: int
    context_length: int = 32
    num_layers: int = 2
    num_heads: int = 4
    model_width: int = 32
    mlp_width: int = 128
    layer_norm_epsilon: float = 1e-5
    initialization_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model_width <= 0 or self.model_width % self.num_heads != 0:
            raise ValueError("model_width must be positive and divisible by num_heads")
        if self.mlp_width <= 0:
            raise ValueError("mlp_width must be positive")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.initialization_std <= 0:
            raise ValueError("initialization_std must be positive")

    @property
    def head_width(self) -> int:
        return self.model_width // self.num_heads

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "GPTConfig":
        return cls(**state)


class GPTModel:
    """A pre-LayerNorm causal transformer with learned token and position embeddings."""

    def __init__(self, config: GPTConfig, seed: int = 0) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        self._parameters: dict[str, Tensor] = {}

        def add_weight(name: str, shape: tuple[int, ...]) -> None:
            values = rng.normal(0.0, config.initialization_std, size=shape).astype(np.float32)
            self._parameters[name] = Tensor(values)

        def add_bias(name: str, width: int) -> None:
            self._parameters[name] = Tensor(np.zeros(width, dtype=np.float32))

        def add_layer_norm(name: str) -> None:
            self._parameters[f"{name}.weight"] = Tensor(np.ones(config.model_width, dtype=np.float32))
            self._parameters[f"{name}.bias"] = Tensor(np.zeros(config.model_width, dtype=np.float32))

        add_weight("token_embedding", (config.vocab_size, config.model_width))
        add_weight("position_embedding", (config.context_length, config.model_width))
        for layer in range(config.num_layers):
            prefix = f"blocks.{layer}"
            add_layer_norm(f"{prefix}.attention_norm")
            for projection in ("query", "key", "value", "output"):
                add_weight(f"{prefix}.{projection}.weight", (config.model_width, config.model_width))
                add_bias(f"{prefix}.{projection}.bias", config.model_width)
            add_layer_norm(f"{prefix}.mlp_norm")
            add_weight(f"{prefix}.mlp_up.weight", (config.model_width, config.mlp_width))
            add_bias(f"{prefix}.mlp_up.bias", config.mlp_width)
            add_weight(f"{prefix}.mlp_down.weight", (config.mlp_width, config.model_width))
            add_bias(f"{prefix}.mlp_down.bias", config.model_width)
        add_layer_norm("final_norm")
        add_weight("lm_head.weight", (config.model_width, config.vocab_size))
        add_bias("lm_head.bias", config.vocab_size)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.data.size for parameter in self._parameters.values())

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        return iter(self._parameters.items())

    def parameters(self) -> Iterator[Tensor]:
        for _, parameter in self.named_parameters():
            yield parameter

    def _parameter(self, name: str) -> Tensor:
        return self._parameters[name]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, input_ids: np.ndarray) -> Tensor:
        """Return logits shaped ``(batch, time, vocabulary)``."""
        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2 or not np.issubdtype(input_ids.dtype, np.integer):
            raise ValueError("input_ids must be a two-dimensional integer array")
        batch_size, sequence_length = input_ids.shape
        if sequence_length <= 0 or sequence_length > self.config.context_length:
            raise ValueError("input sequence length must fit the configured context length")
        if input_ids.size and (input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size):
            raise IndexError("input token is outside the model vocabulary")

        positions = np.arange(sequence_length, dtype=np.int64)
        x = embedding(self._parameter("token_embedding"), input_ids)
        x = x + embedding(self._parameter("position_embedding"), positions)
        for layer in range(self.config.num_layers):
            prefix = f"blocks.{layer}"
            attention_input = layer_norm(
                x,
                self._parameter(f"{prefix}.attention_norm.weight"),
                self._parameter(f"{prefix}.attention_norm.bias"),
                self.config.layer_norm_epsilon,
            )
            query = self._project_heads(attention_input, prefix, "query", batch_size, sequence_length)
            key = self._project_heads(attention_input, prefix, "key", batch_size, sequence_length)
            value = self._project_heads(attention_input, prefix, "value", batch_size, sequence_length)
            attended = causal_attention(query, key, value)
            attended = attended.transpose((0, 2, 1, 3)).reshape(
                (batch_size, sequence_length, self.config.model_width)
            )
            x = x + linear(
                attended,
                self._parameter(f"{prefix}.output.weight"),
                self._parameter(f"{prefix}.output.bias"),
            )

            mlp_input = layer_norm(
                x,
                self._parameter(f"{prefix}.mlp_norm.weight"),
                self._parameter(f"{prefix}.mlp_norm.bias"),
                self.config.layer_norm_epsilon,
            )
            hidden = gelu(
                linear(
                    mlp_input,
                    self._parameter(f"{prefix}.mlp_up.weight"),
                    self._parameter(f"{prefix}.mlp_up.bias"),
                )
            )
            x = x + linear(
                hidden,
                self._parameter(f"{prefix}.mlp_down.weight"),
                self._parameter(f"{prefix}.mlp_down.bias"),
            )

        x = layer_norm(
            x,
            self._parameter("final_norm.weight"),
            self._parameter("final_norm.bias"),
            self.config.layer_norm_epsilon,
        )
        return linear(
            x,
            self._parameter("lm_head.weight"),
            self._parameter("lm_head.bias"),
        )

    def _project_heads(
        self,
        x: Tensor,
        prefix: str,
        projection: str,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        projected = linear(
            x,
            self._parameter(f"{prefix}.{projection}.weight"),
            self._parameter(f"{prefix}.{projection}.bias"),
        )
        return projected.reshape(
            (batch_size, sequence_length, self.config.num_heads, self.config.head_width)
        ).transpose((0, 2, 1, 3))

    def loss(self, input_ids: np.ndarray, targets: np.ndarray) -> Tensor:
        targets = np.asarray(targets)
        if targets.shape != np.asarray(input_ids).shape:
            raise ValueError("targets must have the same shape as input_ids")
        return cross_entropy(self.forward(input_ids), targets)

    def save(self, path: Path) -> None:
        """Save the model config and parameters to a NumPy checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, object] = {
            "config_json": np.asarray(json.dumps(self.config.to_dict())),
        }
        for index, (_, parameter) in enumerate(self.named_parameters()):
            values[f"parameter_{index:04d}"] = parameter.data
        np.savez_compressed(path, **values)

    @classmethod
    def load(cls, path: Path) -> "GPTModel":
        """Load a model checkpoint and restore parameters exactly."""
        with np.load(path, allow_pickle=False) as archive:
            config_state = json.loads(str(archive["config_json"].item()))
            model = cls(GPTConfig.from_dict(config_state), seed=0)
            for index, (_, parameter) in enumerate(model.named_parameters()):
                key = f"parameter_{index:04d}"
                if key not in archive:
                    raise ValueError(f"Checkpoint is missing {key}")
                values = archive[key]
                if values.shape != parameter.shape:
                    raise ValueError(f"Checkpoint shape for {key} does not match the model")
                parameter.data[...] = values
        return model
