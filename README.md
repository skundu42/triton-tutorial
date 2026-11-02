# Triton From Scratch

A hands-on, five-part beginner tutorial: from basic PyTorch to writing and optimizing a real LLM kernel in
[Triton](https://github.com/triton-lang/triton), OpenAI's Python language and compiler for GPU kernels. Each part is one
self-contained script you can read top to bottom and run.

| Part | File | Kernel | You learn |
|---|---|---|---|
| 1 | `01-vector-add.py` | vector addition | programs, grid, blocks, offsets, masks, `tl.load` / `tl.store`, launching, `do_bench` |
| 2 | `02-fused-softmax.py` | fused softmax | memory traffic, kernel fusion, reductions (`tl.max`, `tl.sum`), power-of-two blocks, numerical stability |
| 3 | `03-matmul.py` | tiled matrix multiply | 2-D indexing and strides, tiling, `tl.dot` on Tensor Cores, fp32 accumulators, grouped (L2-friendly) ordering |
| 4 | `04-optimized-matmul.py` | tuned matmul | tile sizes, `num_warps`, `num_stages`, `@triton.autotune`, memory-bound vs compute-bound |
| 5 | `05-rmsnorm.py` | RMSNorm (LLM layer) | everything above in one real kernel, validation, a drop-in `nn.Module` |

## Setup

```bash
pip install torch          # on Linux, PyTorch's CUDA wheels already install triton
python 01-vector-add.py
```

You need an NVIDIA GPU with compute capability 8.0 or newer (RTX 30/40/50-series, A100, H100, ...).

No GPU? Triton's interpreter runs kernels on the CPU with NumPy (slow, but great for learning and debugging; Linux only,
since Triton has no macOS wheels):

```bash
TRITON_INTERPRET=1 python 01-vector-add.py
```

Parts 2 to 5 use `device="cuda"` throughout; switch the `DEVICE` constant to `"cpu"` to try them in the interpreter
(bfloat16 is not supported there, so use float32 for part 5).

## Notes

- Reference numbers to compare against: Triton's own tutorial outputs (built nightly on an NVIDIA A100) for parts 1 to
  4, and Liger-Kernel's published benchmark data for part 5. Your GPU will give different numbers: run the files and
  compare.
- `04-optimized-matmul.py` uses the first eight CUDA configs from Triton's official matmul tutorial. Run it with
  `TRITON_PRINT_AUTOTUNING=1` to see which config wins on your GPU.
- `05-rmsnorm.py` is forward-only (inference). Training needs a backward kernel; see
  [Liger Kernel](https://github.com/linkedin/Liger-Kernel) for a production version.

Further reading: the official [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
and [Triton Puzzles](https://github.com/srush/Triton-Puzzles).
