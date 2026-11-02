"""Triton From Scratch, part 5: a real LLM kernel, a fused RMSNorm (forward pass, for inference).

Run:  python 05-rmsnorm.py
"""
import torch
import triton
import triton.language as tl

DEVICE = "cuda"


def torch_rmsnorm(x, weight, eps=1e-6):
    """The eager PyTorch version found in many model files (e.g. Hugging Face's Llama/Qwen RMSNorm)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, x_stride, out_stride, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                               # one program per token (row)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * x_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols            # padding is 0, so it adds nothing
    rstd = tl.rsqrt(variance + eps)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    out = x * rstd * w                                   # stored back as the input dtype
    tl.store(out_ptr + row * out_stride + offsets, out, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(-1, shape[-1])                         # (tokens, hidden)
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 16 if BLOCK_SIZE >= 8192 else 8 if BLOCK_SIZE >= 2048 else 4   # wider rows, more threads
    rmsnorm_kernel[(n_rows,)](x, weight, out, x.stride(0), out.stride(0), n_cols, eps,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out.reshape(shape)


class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for a model's RMSNorm module (inference only: no backward kernel)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)


if __name__ == "__main__":
    torch.manual_seed(0)
    for hidden in (4096, 8192, 16384):
        x = torch.randn(2048, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.rand(hidden, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(rmsnorm(x, w), torch_rmsnorm(x, w))
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms_tri = triton.testing.do_bench(lambda: rmsnorm(x, w))
        ms_eager = triton.testing.do_bench(lambda: torch_rmsnorm(x, w))
        ms_nn = triton.testing.do_bench(lambda: torch.nn.functional.rms_norm(x, (hidden,), w, 1e-6))
        print(f"hidden {hidden:>5}: triton {ms_tri * 1e3:6.1f} us ({gbps(ms_tri):6.0f} GB/s)   "
              f"eager {ms_eager * 1e3:6.1f} us   F.rms_norm {ms_nn * 1e3:6.1f} us")
