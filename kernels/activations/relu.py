"""
ReLU activation kernel implemented in Triton.

Formula: relu(x) = max(x, 0)

This is the simplest possible Triton kernel — a single elementwise op.
It's the foundation for understanding:
  - how Triton programs are structured
  - how blocks/threads map to data
  - how to verify correctness against PyTorch
"""

import triton
import triton.language as tl


@triton.jit
def relu_forward_kernel(
    x_ptr,        # pointer to input tensor
    out_ptr,      # pointer to output tensor
    N,            # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements each program handles
):
    # Each program instance handles a block of BLOCK_SIZE elements.
    # program_id(0) gives this instance's index along axis 0.
    pid = tl.program_id(axis=0)

    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE

    # Create a range of offsets: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out-of-bounds memory accesses (last block may be partial)
    mask = offsets < N

    # Load input values from global memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)

    # Store result back to global memory
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def relu_backward_kernel(
    x_ptr,        # pointer to original input (saved from forward)
    grad_out_ptr, # pointer to upstream gradient
    grad_in_ptr,  # pointer to output gradient
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)

    # Gradient of ReLU: pass through grad where x > 0, else 0
    grad_in = tl.where(x > 0, grad_out, 0.0)

    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)