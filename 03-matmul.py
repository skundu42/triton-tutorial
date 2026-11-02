"""Triton From Scratch, part 3: a tiled matrix multiplication (FP16 in, FP32 accumulate).

Run:  python 03-matmul.py
"""
import torch
import triton
import triton.language as tl

DEVICE = "cuda"


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, ACTIVATION: tl.constexpr):
    # 1. which tile of C does this program own?
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # simple row-major order would be:  pid_m = pid // num_pid_n ; pid_n = pid % num_pid_n
    # grouped order walks GROUP_M tile-rows together, so neighbouring programs reuse tiles in L2 cache
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. pointers to the first K-slice of A's row-tile and B's column-tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 3. walk along K, multiplying tiles on the Tensor Cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. optional fused epilogue: applied while the tile is still in registers
    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)

    # 5. write the finished C tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "incompatible shapes"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, ACTIVATION=activation)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    print(torch.allclose(matmul(a, b), a @ b, atol=1e-2, rtol=0))
    print(torch.allclose(matmul(a, b, activation="leaky_relu"),
                         torch.nn.functional.leaky_relu(a @ b), atol=1e-2, rtol=0))

    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
        b = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
        tflops = lambda ms: 2 * n**3 * 1e-12 / (ms * 1e-3)
        ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
        ms_torch = triton.testing.do_bench(lambda: a @ b)
        print(f"{n}^3   triton {tflops(ms_triton):6.1f} TFLOPS   cuBLAS {tflops(ms_torch):6.1f} TFLOPS")
