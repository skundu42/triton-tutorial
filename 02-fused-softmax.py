"""Triton From Scratch, part 2: a fused softmax (one program per row).

Run:  python 02-fused-softmax.py
"""
import torch
import triton
import triton.language as tl

DEVICE = "cuda"


def naive_softmax(x):
    """Five separate PyTorch ops: each one reads its input from GPU memory and writes its result back."""
    x_max = x.max(dim=1, keepdim=True).values
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                     # one program per row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(in_ptr + row_idx * in_stride + offsets,
                  mask=mask, other=-float("inf"))  # padding can never win the max
    row_minus_max = row - tl.max(row, axis=0)      # numerical stability
    numerator = tl.exp(row_minus_max)              # exp(-inf) = 0 for the padding
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    tl.store(out_ptr + row_idx * out_stride + offsets, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)    # blocks must be a power of two
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)      # irregular shape: 781 columns -> BLOCK_SIZE 1024
    print(torch.allclose(softmax(x), torch.softmax(x, dim=1), atol=1e-6))

    # bandwidth: the fused kernel reads x once and writes y once
    for n_cols in (256, 1024, 4096, 8192):
        x = torch.randn(4096, n_cols, device=DEVICE)
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        row = [f"N={n_cols:>5}"]
        for name, fn in (("triton", softmax), ("torch", lambda t: torch.softmax(t, dim=-1)),
                         ("naive", naive_softmax)):
            ms = triton.testing.do_bench(lambda: fn(x))
            row.append(f"{name} {gbps(ms):7.1f} GB/s")
        print("   ".join(row))
