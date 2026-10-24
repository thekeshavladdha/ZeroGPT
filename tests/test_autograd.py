import unittest

import numpy as np

from zerogpt.autograd import (
    Tensor,
    causal_attention,
    cross_entropy,
    embedding,
    gelu,
    layer_norm,
    linear,
    softmax,
)


def check_finite_difference(
    testcase: unittest.TestCase,
    loss_function,
    tensors: list[Tensor],
    epsilon: float = 1e-3,
    absolute_tolerance: float = 2e-3,
    relative_tolerance: float = 2e-2,
) -> None:
    for tensor in tensors:
        tensor.zero_grad()
    loss = loss_function()
    loss.backward()

    for tensor in tensors:
        analytical = tensor.grad.copy()
        for index in np.ndindex(tensor.shape):
            original = float(tensor.data[index])
            tensor.data[index] = original + epsilon
            above = float(loss_function().data)
            tensor.data[index] = original - epsilon
            below = float(loss_function().data)
            tensor.data[index] = original
            numerical = (above - below) / (2.0 * epsilon)
            tolerance = absolute_tolerance + relative_tolerance * abs(numerical)
            testcase.assertLessEqual(
                abs(float(analytical[index]) - numerical),
                tolerance,
                msg=f"gradient mismatch for {tensor.shape} at {index}: "
                f"analytical={analytical[index]}, numerical={numerical}",
            )


def weighted_sum(tensor: Tensor) -> Tensor:
    weights = np.linspace(0.2, 1.1, tensor.data.size, dtype=np.float32).reshape(tensor.shape)
    return (tensor * weights).sum()


class AutogradGradientTests(unittest.TestCase):
    def test_broadcast_add_and_multiply(self) -> None:
        left = Tensor([[0.2, -0.4, 0.7], [0.1, 0.8, -0.3]])
        right = Tensor([0.5, -0.2, 0.9])
        check_finite_difference(self, lambda: weighted_sum(left + right), [left, right])

        left = Tensor([[0.2, -0.4, 0.7], [0.1, 0.8, -0.3]])
        right = Tensor([0.5, 0.8, 1.2])
        check_finite_difference(self, lambda: weighted_sum(left * right), [left, right])

    def test_division_power_and_reductions(self) -> None:
        numerator = Tensor([[0.5, 1.2], [1.7, 0.8]])
        denominator = Tensor([[1.1, 0.7]])
        check_finite_difference(
            self, lambda: weighted_sum(numerator / denominator), [numerator, denominator]
        )

        x = Tensor([[0.4, 0.8], [1.2, 1.5]])
        check_finite_difference(self, lambda: weighted_sum(x**2.3), [x])
        check_finite_difference(self, lambda: x.sum(axis=0).mean(), [x])

    def test_matmul_reshape_transpose_and_linear(self) -> None:
        x = Tensor([[0.2, -0.3, 0.6], [0.7, 0.1, -0.2]])
        weight = Tensor([[0.4, -0.5], [0.8, 0.2], [-0.1, 0.6]])
        bias = Tensor([0.1, -0.2])
        check_finite_difference(self, lambda: weighted_sum(linear(x, weight, bias)), [x, weight, bias])

        x = Tensor([[0.3, -0.2, 0.8], [1.0, 0.4, -0.5]])
        check_finite_difference(
            self,
            lambda: x.transpose((1, 0)).reshape((6,)).mean(),
            [x],
        )

    def test_embedding_with_repeated_indices(self) -> None:
        weight = Tensor([[0.2, -0.3], [0.7, 0.4], [-0.2, 0.9]])
        ids = np.array([[1, 1], [0, 2]], dtype=np.int64)
        check_finite_difference(self, lambda: weighted_sum(embedding(weight, ids)), [weight])

    def test_gelu_and_softmax(self) -> None:
        x = Tensor([[-1.1, -0.2], [0.4, 1.3]])
        check_finite_difference(self, lambda: weighted_sum(gelu(x)), [x])

        x = Tensor([[0.2, -0.7, 1.0], [0.5, 0.8, -0.3]])
        check_finite_difference(self, lambda: weighted_sum(softmax(x)), [x])

    def test_layer_norm(self) -> None:
        x = Tensor([[0.2, -0.7, 1.1], [1.4, 0.3, -0.2]])
        weight = Tensor([0.8, 1.1, 0.9])
        bias = Tensor([0.1, -0.2, 0.05])
        check_finite_difference(self, lambda: weighted_sum(layer_norm(x, weight, bias)), [x, weight, bias])

    def test_cross_entropy(self) -> None:
        logits = Tensor([[0.3, -0.1, 0.8], [-0.6, 0.5, 0.2]])
        targets = np.array([2, 1], dtype=np.int64)
        check_finite_difference(self, lambda: cross_entropy(logits, targets), [logits])

    def test_causal_attention(self) -> None:
        rng = np.random.default_rng(12)
        query = Tensor(rng.normal(size=(1, 1, 2, 2)))
        key = Tensor(rng.normal(size=(1, 1, 2, 2)))
        value = Tensor(rng.normal(size=(1, 1, 2, 2)))
        weights = Tensor(np.array([[[[0.3, 0.8], [-0.2, 1.1]]]], dtype=np.float32), requires_grad=False)
        check_finite_difference(
            self,
            lambda: (causal_attention(query, key, value) * weights).sum(),
            [query, key, value],
        )

    def test_causal_attention_ignores_future_inputs(self) -> None:
        query = Tensor(np.ones((1, 1, 3, 2), dtype=np.float32))
        key = Tensor(np.zeros((1, 1, 3, 2), dtype=np.float32))
        value = Tensor(np.array([[[[1, 0], [0, 1], [10, 10]]]], dtype=np.float32))
        original = causal_attention(query, key, value).data

        changed_query = query.data.copy()
        changed_query[:, :, 2, :] = 100.0
        changed_key = key.data.copy()
        changed_key[:, :, 2, :] = -100.0
        changed_value = value.data.copy()
        changed_value[:, :, 2, :] = -100.0
        changed = causal_attention(
            Tensor(changed_query), Tensor(changed_key), Tensor(changed_value)
        ).data

        np.testing.assert_allclose(original[:, :, :2], changed[:, :, :2], atol=1e-6)

    def test_tiny_fixed_batch_can_be_overfit(self) -> None:
        inputs = Tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]], requires_grad=False)
        targets = np.array([[0], [1], [0], [1]], dtype=np.int64)
        weight = Tensor([[0.1, -0.1], [-0.2, 0.2]])
        bias = Tensor([0.0, 0.0])

        for _ in range(120):
            weight.zero_grad()
            bias.zero_grad()
            loss = cross_entropy(linear(inputs, weight, bias), targets)
            loss.backward()
            weight.data -= 0.5 * weight.grad
            bias.data -= 0.5 * bias.grad

        final_loss = cross_entropy(linear(inputs, weight, bias), targets)
        self.assertLess(float(final_loss.data), 0.02)


if __name__ == "__main__":
    unittest.main()
