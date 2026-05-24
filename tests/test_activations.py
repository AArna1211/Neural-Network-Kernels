"""
tests/test_activations.py

Correctness tests for custom activation kernels.

Two tests per kernel:
  1. allclose  — output matches PyTorch numerically
  2. gradcheck — gradients match numerical approximation

Run with:
  pytest tests/test_activations.py -v
"""

import torch
import pytest
from torch.autograd import gradcheck
from ops.activations import relu, ReLUFunction


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ── Forward correctness ───────────────────────────────────────────────────────

class TestReLUForward:

    def test_basic_correctness(self, device):
        """Custom ReLU must match torch.relu exactly."""
        x = torch.randn(1024, device=device)
        expected = torch.relu(x)
        actual = relu(x)
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"Max diff: {(actual - expected).abs().max()}"

    def test_large_tensor(self, device):
        """Works on large tensors (tests block boundary handling)."""
        x = torch.randn(10_000_000, device=device)
        expected = torch.relu(x)
        actual = relu(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_negative_values_zeroed(self, device):
        """All negative inputs must produce exactly 0."""
        x = -torch.abs(torch.randn(1024, device=device)) - 1e-6
        out = relu(x)
        assert (out == 0).all(), "Negative values should be zeroed"

    def test_positive_values_unchanged(self, device):
        """All positive inputs must pass through unchanged."""
        x = torch.abs(torch.randn(1024, device=device)) + 1e-6
        out = relu(x)
        assert torch.allclose(out, x, atol=1e-6), \
            "Positive values should be unchanged"

    def test_multidimensional(self, device):
        """Works on non-flat tensors."""
        x = torch.randn(32, 64, 128, device=device)
        expected = torch.relu(x)
        actual = relu(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_non_block_aligned_size(self, device):
        """Works when tensor size isn't a multiple of BLOCK_SIZE (1024)."""
        x = torch.randn(1337, device=device)  # odd size
        expected = torch.relu(x)
        actual = relu(x)
        assert torch.allclose(actual, expected, atol=1e-6)


# ── Backward correctness ──────────────────────────────────────────────────────

class TestReLUBackward:

    def test_gradcheck(self, device):
        """
        gradcheck numerically estimates gradients and compares to your
        backward kernel. Uses float64 for numerical stability.
        If this passes, your backward is correct.
        """
        x = torch.randn(64, dtype=torch.float64, device=device,
                        requires_grad=True)
        assert gradcheck(ReLUFunction.apply, (x,), eps=1e-4, atol=1e-4), \
            "gradcheck failed — backward kernel has a bug"

    def test_gradient_values(self, device):
        """Gradient is 1 where x > 0, 0 where x <= 0."""
        x = torch.randn(1024, device=device, requires_grad=True)
        out = relu(x)
        out.sum().backward()

        expected_grad = (x > 0).float()
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
            f"Gradient mismatch. Max diff: {(x.grad - expected_grad).abs().max()}"

    def test_gradient_flow(self, device):
        """Gradient flows correctly through a simple computation graph."""
        x = torch.randn(256, device=device, requires_grad=True)
        out = relu(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ── Quick sanity check (run directly) ────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.randn(1024, device=device)

    out_custom = relu(x)
    out_torch  = torch.relu(x)

    match = torch.allclose(out_custom, out_torch, atol=1e-6)
    print(f"Forward correct: {match}")
    print(f"Max diff:        {(out_custom - out_torch).abs().max():.2e}")