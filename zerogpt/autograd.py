"""Minimal NumPy autograd and operations required by the tiny GPT."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


Backward = Callable[[np.ndarray], None]


def _sum_to_shape(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting by summing added and expanded dimensions."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """A float32 NumPy array with a small reverse-mode computation graph."""

    def __init__(self, data: object, requires_grad: bool = True) -> None:
        self.data = np.array(data, dtype=np.float32, copy=True)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._prev: tuple[Tensor, ...] = ()
        self._backward: Backward = lambda _gradient: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def _coerce(value: Tensor | object) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)

    @staticmethod
    def _operation(
        data: np.ndarray | float,
        parents: Sequence[Tensor],
        backward: Backward,
    ) -> Tensor:
        result = Tensor(data, requires_grad=any(parent.requires_grad for parent in parents))
        result._prev = tuple(parent for parent in parents if parent.requires_grad)
        result._backward = backward
        return result

    def _add_grad(self, gradient: np.ndarray) -> None:
        if not self.requires_grad:
            return
        gradient = np.asarray(gradient, dtype=np.float32)
        if gradient.shape != self.shape:
            raise ValueError(f"Gradient shape {gradient.shape} does not match tensor shape {self.shape}")
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += gradient

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, gradient: np.ndarray | float | None = None) -> None:
        if not self.requires_grad:
            raise ValueError("Cannot backpropagate from a tensor that does not require gradients")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("A gradient must be supplied for a non-scalar output")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=np.float32)
            if seed.shape != self.shape:
                raise ValueError(f"Output gradient shape {seed.shape} does not match {self.shape}")

        order: list[Tensor] = []
        visited: set[Tensor] = set()

        def visit(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    visit(parent)
                order.append(node)

        visit(self)
        for node in order:
            if node._prev:
                node.grad = None
        self._add_grad(seed)
        for node in reversed(order):
            if node.grad is not None:
                node._backward(node.grad)

    def __add__(self, other: Tensor | object) -> Tensor:
        other = self._coerce(other)
        left_shape, right_shape = self.shape, other.shape

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(_sum_to_shape(gradient, left_shape))
            other._add_grad(_sum_to_shape(gradient, right_shape))

        return self._operation(self.data + other.data, (self, other), backward)

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | object) -> Tensor:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Tensor | object) -> Tensor:
        return self._coerce(other) - self

    def __mul__(self, other: Tensor | object) -> Tensor:
        other = self._coerce(other)
        left_shape, right_shape = self.shape, other.shape

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(_sum_to_shape(gradient * other.data, left_shape))
            other._add_grad(_sum_to_shape(gradient * self.data, right_shape))

        return self._operation(self.data * other.data, (self, other), backward)

    __rmul__ = __mul__

    def __truediv__(self, other: Tensor | object) -> Tensor:
        other = self._coerce(other)
        left_shape, right_shape = self.shape, other.shape

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(_sum_to_shape(gradient / other.data, left_shape))
            other._add_grad(
                _sum_to_shape(-gradient * self.data / (other.data * other.data), right_shape)
            )

        return self._operation(self.data / other.data, (self, other), backward)

    def __rtruediv__(self, other: Tensor | object) -> Tensor:
        return self._coerce(other) / self

    def __pow__(self, power: float) -> Tensor:
        if not isinstance(power, (int, float)):
            raise TypeError("Tensor powers must be numeric scalars")

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(gradient * power * np.power(self.data, power - 1))

        return self._operation(np.power(self.data, power), (self,), backward)

    def __matmul__(self, other: Tensor | object) -> Tensor:
        other = self._coerce(other)
        left_shape, right_shape = self.shape, other.shape

        def backward(gradient: np.ndarray) -> None:
            left_gradient = np.matmul(gradient, np.swapaxes(other.data, -1, -2))
            right_gradient = np.matmul(np.swapaxes(self.data, -1, -2), gradient)
            self._add_grad(_sum_to_shape(left_gradient, left_shape))
            other._add_grad(_sum_to_shape(right_gradient, right_shape))

        return self._operation(self.data @ other.data, (self, other), backward)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        value = self.data.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            axes = tuple(range(self.data.ndim))
        elif isinstance(axis, int):
            axes = (axis % self.data.ndim,)
        else:
            axes = tuple(item % self.data.ndim for item in axis)

        def backward(gradient: np.ndarray) -> None:
            expanded = gradient
            if not keepdims:
                for item in sorted(axes):
                    expanded = np.expand_dims(expanded, axis=item)
            self._add_grad(np.broadcast_to(expanded, self.shape))

        return self._operation(value, (self,), backward)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[item] for item in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, shape: tuple[int, ...] | int) -> Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        original_shape = self.shape

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(gradient.reshape(original_shape))

        return self._operation(self.data.reshape(shape), (self,), backward)

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))
        axes = tuple(item % self.data.ndim for item in axes)
        inverse = tuple(np.argsort(axes))

        def backward(gradient: np.ndarray) -> None:
            self._add_grad(gradient.transpose(inverse))

        return self._operation(self.data.transpose(axes), (self,), backward)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def embedding(weight: Tensor, indices: np.ndarray) -> Tensor:
    """Gather rows from a vocabulary-by-width weight matrix."""
    indices = np.asarray(indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Embedding indices must be integers")
    if indices.size and (indices.min() < 0 or indices.max() >= weight.shape[0]):
        raise IndexError("Embedding index is outside the weight matrix")

    def backward(gradient: np.ndarray) -> None:
        weight_gradient = np.zeros_like(weight.data)
        np.add.at(weight_gradient, indices, gradient)
        weight._add_grad(weight_gradient)

    return Tensor._operation(weight.data[indices], (weight,), backward)


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    result = x @ weight
    return result if bias is None else result + bias


def gelu(x: Tensor) -> Tensor:
    """GELU with the tanh approximation used by small GPT implementations."""
    coefficient = np.float32(np.sqrt(2.0 / np.pi))
    cubic = np.float32(0.044715)
    inner = coefficient * (x.data + cubic * x.data**3)
    tangent = np.tanh(inner)
    value = 0.5 * x.data * (1.0 + tangent)

    def backward(gradient: np.ndarray) -> None:
        derivative = 0.5 * (1.0 + tangent) + 0.5 * x.data * (1.0 - tangent**2) * coefficient * (
            1.0 + 3.0 * cubic * x.data**2
        )
        x._add_grad(gradient * derivative)

    return Tensor._operation(value, (x,), backward)


def layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """Normalize the final dimension, then apply learned scale and offset."""
    if epsilon <= 0:
        raise ValueError("LayerNorm epsilon must be positive")
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    normalized = centered / (variance + epsilon) ** 0.5
    return normalized * weight + bias


def softmax(x: Tensor, axis: int = -1, mask: np.ndarray | None = None) -> Tensor:
    """Numerically stable softmax, optionally excluding masked positions."""
    axis %= x.data.ndim
    if mask is None:
        allowed = np.ones(x.shape, dtype=bool)
    else:
        allowed = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
        if np.any(~allowed.any(axis=axis)):
            raise ValueError("Every softmax row must contain at least one unmasked value")

    masked = np.where(allowed, x.data, -np.inf)
    shifted = masked - np.max(masked, axis=axis, keepdims=True)
    exponentials = np.where(allowed, np.exp(shifted), 0.0)
    probabilities = exponentials / exponentials.sum(axis=axis, keepdims=True)

    def backward(gradient: np.ndarray) -> None:
        dot = (gradient * probabilities).sum(axis=axis, keepdims=True)
        x_gradient = probabilities * (gradient - dot)
        x._add_grad(np.where(allowed, x_gradient, 0.0))

    return Tensor._operation(probabilities, (x,), backward)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy over the leading dimensions."""
    targets = np.asarray(targets)
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("Cross-entropy targets must be integer class indices")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"Target shape {targets.shape} does not match logits {logits.shape}")

    vocab_size = logits.shape[-1]
    flat_logits = logits.data.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1).astype(np.int64, copy=False)
    if flat_targets.size == 0:
        raise ValueError("Cross-entropy requires at least one target")
    if flat_targets.min() < 0 or flat_targets.max() >= vocab_size:
        raise IndexError("Cross-entropy target is outside the vocabulary")

    row_max = flat_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(flat_logits - row_max)
    normalizers = exponentials.sum(axis=1, keepdims=True)
    probabilities = exponentials / normalizers
    rows = np.arange(flat_targets.size)
    losses = np.log(normalizers[:, 0]) + row_max[:, 0] - flat_logits[rows, flat_targets]
    value = np.asarray(losses.mean(), dtype=np.float32)

    def backward(gradient: np.ndarray) -> None:
        logits_gradient = probabilities.copy()
        logits_gradient[rows, flat_targets] -= 1.0
        logits_gradient *= gradient / flat_targets.size
        logits._add_grad(logits_gradient.reshape(logits.shape))

    return Tensor._operation(value, (logits,), backward)


def causal_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Scaled causal attention for tensors shaped (batch, heads, time, width)."""
    if query.shape != key.shape or query.shape != value.shape or len(query.shape) != 4:
        raise ValueError("Query, key, and value must share shape (batch, heads, time, width)")
    _, _, sequence_length, head_width = query.shape
    causal_mask = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))
    scores = (query @ key.transpose((0, 1, 3, 2))) / np.sqrt(head_width)
    probabilities = softmax(scores, axis=-1, mask=causal_mask)
    return probabilities @ value
