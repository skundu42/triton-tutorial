"""Triton From Scratch, part 1: your first GPU kernel (vector addition).

Run:  python 01-vector-add.py          (NVIDIA GPU, compute capability 8.0+)
      TRITON_INTERPRET=1 python 01-vector-add.py   (no GPU: simulated on the CPU with NumPy, slow)
"""
import os

import torch
import triton
import triton.language as tl

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                            # which program am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # the elements this program owns
    mask = offsets < n_elements                            # guard the ragged last block
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_relu_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    # the same kernel with one changed line: relu(x + y) in a single pass (kernel fusion)
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x + y, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_relu_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    # 1. correctness: an awkward size, so the last block is ragged and the mask is exercised
    torch.manual_seed(0)
    x = torch.rand(98_432, device=DEVICE)
    y = torch.rand(98_432, device=DEVICE)
    output = add(x, y)
    print(torch.allclose(output, x + y))
    print((output - (x + y)).abs().max())

    # 2. speed: bandwidth in GB/s (reads x and y, writes output: 3 x 4 bytes per element)
    if DEVICE == "cuda":
        for size in (2**12, 2**20, 2**27):
            x = torch.rand(size, device=DEVICE)
            y = torch.rand(size, device=DEVICE)
            gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
            ms_triton = triton.testing.do_bench(lambda: add(x, y))
            ms_torch = triton.testing.do_bench(lambda: x + y)
            print(f"{size:>11,}  triton {gbps(ms_triton):7.1f} GB/s   torch {gbps(ms_torch):7.1f} GB/s")

        # 3. fusion: relu(x + y). PyTorch runs two kernels (20 bytes/element), ours runs one (12 bytes/element)
        assert torch.allclose(add_relu(x, y), torch.relu(x + y))
        ms_fused = triton.testing.do_bench(lambda: add_relu(x, y))
        ms_torch2 = triton.testing.do_bench(lambda: torch.relu(x + y))
        print(f"relu(x + y): fused triton {ms_fused:.3f} ms   torch {ms_torch2:.3f} ms   "
              f"speed-up {ms_torch2 / ms_fused:.2f}x")
