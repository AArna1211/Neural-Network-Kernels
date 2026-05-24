"""
ops/activations.py

Wraps the raw Triton kernels in torch.autograd.Function so PyTorch's
autograd engine can call them during forward and backward passes.

This is the pattern you'll use for every kernel in this project:
  kernel (kernels/) → autograd wrapper (ops/) → model usage (model/)
"""

import torch
import triton
from kernels.activations.relu import relu_forward_kernel, relu_backward_kernel


class ReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # x must be contiguous in memory for pointer arithmetic to work
        x = x.contiguous()
        out = torch.empty_like(x)

        N = x.numel()
        BLOCK_SIZE = 1024  # process 1024 elements per program instance

        # Grid = number of blocks needed to cover all N elements
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        relu_forward_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save input for backward pass
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(x)

        N = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        relu_backward_kernel[grid](
            x_ptr=x,
            grad_out_ptr=grad_output,
            grad_in_ptr=grad_input,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_input


def relu(x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.relu, backed by your Triton kernel."""
    return ReLUFunction.apply(x)