# Neural-Network-Kernels

Small collection of Triton-backed kernels and PyTorch wrappers used
to explore high-performance custom operators for neural networks.

**Progress so far**

- Implemented a Triton ReLU kernel with both forward and backward
	implementations: [kernels/activations/relu.py](kernels/activations/relu.py).
- Wrapped the raw Triton kernels in a PyTorch `autograd.Function`
	and exposed a drop-in `relu(x)` in [ops/activations.py](ops/activations.py).
- Added comprehensive correctness tests in
	[tests/test_activations.py](tests/test_activations.py) (forward checks
	and `gradcheck` for the backward pass).

**Why this repo exists**

This repository is an experimental playground for writing custom
GPU kernels in Triton and integrating them with PyTorch. The goal is to
learn kernel structure, memory handling, and how to expose kernels
safely to PyTorch users while keeping correct gradients.

**How to run the tests (quick)**

1. Create and activate a Python environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the activation tests (requires CUDA + Triton):

```bash
pytest tests/test_activations.py -v
```

Note: The tests will skip if a CUDA device is not available.

**Implemented components (files)**

- kernels/activations/relu.py — Triton forward & backward kernels
- ops/activations.py — `ReLUFunction` autograd wrapper and `relu()` helper
- tests/test_activations.py — numerical and gradient tests

**Dependencies**

See `requirements.txt` for pinned development/test dependencies. The
core runtime libraries are `torch` and `triton` (both required for
running the kernels on GPU).

**Next steps**

- Add more activation kernels (GELU, SiLU) and corresponding tests.
- Add CI that runs tests on a GPU-enabled runner or uses CPU fallbacks.
- Add benchmarking harnesses to compare performance vs `torch` ops.
