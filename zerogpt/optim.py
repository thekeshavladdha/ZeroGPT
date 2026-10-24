"""NumPy AdamW optimizer and gradient clipping."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .autograd import Tensor


class AdamW:
    """AdamW with global L2 gradient clipping for the model parameters."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        learning_rate: float = 3e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("AdamW requires at least one parameter")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.step_count = 0
        self.first_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.second_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self, learning_rate: float | None = None) -> dict[str, float]:
        """Apply one clipped AdamW update and return gradient statistics.

        ``learning_rate`` overrides the constructor value, which lets a
        scheduler (e.g. cosine decay) drive the rate without rebuilding
        the optimizer state.
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        gradients = [
            parameter.grad if parameter.grad is not None else np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]
        gradient_norm = float(np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients)))
        clip_scale = min(1.0, self.gradient_clip_norm / (gradient_norm + 1e-12))
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        for parameter, gradient, first, second in zip(
            self.parameters, gradients, self.first_moments, self.second_moments
        ):
            clipped = gradient * clip_scale
            first *= self.beta1
            first += (1.0 - self.beta1) * clipped
            second *= self.beta2
            second += (1.0 - self.beta2) * clipped * clipped
            first_hat = first / bias_correction1
            second_hat = second / bias_correction2
            update = first_hat / (np.sqrt(second_hat) + self.epsilon)
            if self.weight_decay:
                update = update + self.weight_decay * parameter.data
            parameter.data -= learning_rate * update
        return {"gradient_norm": gradient_norm, "clip_scale": float(clip_scale)}
