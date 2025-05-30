# fp8-blockwise-gemm-cutlass-hopper

> **Era:** 2025+ · Staff AI Engineer focus area (low-precision inference kernels). Technique from 2024, reimplemented in 2026 with CUTLASS 4.x and PyTorch 2.x.

An independent 128×128 block-scaled FP8 E4M3 quantization/GEMM reference, plus a real CUTLASS/CuTe SM90 executable using TMA, warp specialization and WGMMA block scaling.

## Problem

A single tensor-wise FP8 scale couples the largest and smallest values in a matrix. Block scaling preserves more local dynamic range, but the GEMM must apply the correct pair of scales after every K-block rather than once after the full reduction.

## Method

For an `M×K` activation and `K×N` weight, each 128-row by 128-column block is quantized independently:

```
s[b] = max(abs(X[b])) / 448       Q[b] = fp8_e4m3(clamp(X[b] / s[b]))
C = sum_kb (Qa[:, kb] @ Qb[kb, :]) * (sa[:, kb] @ sb[kb, :])
```

The last product of scale vectors is an outer product. Empty blocks use scale 1. The Python reference zero-pads incomplete edge blocks, stores values as `torch.float8_e4m3fn` and scales as fp32, and accumulates every K-block in fp32. Its `blockwise_gemm` follows the block formula directly rather than replacing the operation with an unquantized matmul.

The CUDA executable instantiates CUTLASS's `KernelTmaWarpSpecializedCooperativeFP8Blockwise` collective with a `128×128×128` tile and `sm90_trivial_blockwise_scale_config`, so its scale granularity matches the reference. CUTLASS's `GettBlockScalingMainloopParams` host reference verifies the GPU output before timing. The benchmark also times cuBLAS BF16 with fp32 accumulation at the same shape; it does not compare accuracy across those different input formats.

## Results

Local CPU correctness smoke, seed 123, `M=130, K=256, N=130`, PyTorch 2.15.0+cpu:

| Metric | Measured |
|---|---:|
| normalized RMSE vs fp32 GEMM | 0.0377047025 |
| maximum absolute error | 2.2093183994 |
| padded FP8 value storage | 131072 bytes |
| fp32 scale storage | 32 bytes |

This is a numerical accuracy result, not a latency benchmark. SM90 compilation, correctness and performance are **pending**: this development machine has no CUDA toolkit or GPU.

| GPU | shape | CUTLASS FP8 TFLOPS | cuBLAS BF16 TFLOPS |
|---|---|---:|---:|
| _pending H100_ | 1024×1024×1024 | pending | pending |

The Hopper command that produces both columns is:

```bash
CUTLASS_DIR=/path/to/cutlass scripts/bench_sm90.sh
```

## Reproduce

CPU tests and deterministic smoke:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest -q
python -m fp8_blockwise.bench --smoke
```

Hopper requires a CUDA 12.8+ toolkit, CMake 3.24+, a CUTLASS 4.3+ checkout and an SM90 GPU:

```bash
CUTLASS_DIR=/path/to/cutlass scripts/build_sm90.sh
build/sm90/fp8_blockwise_sm90 --m=1024 --n=1024 --k=1024 --warmup=10 --iterations=50
# Add --no-verify only for timing after a verified run.
```

No CPU script emits a synthetic GPU timing or claims to have compiled the SM90 source.

## Limitations

- E4M3 is finite-only/saturating here; NaN and infinite source values are rejected.
- The CPU reference pads arbitrary shapes. The standalone CUDA benchmark restricts M, N and K to positive multiples of 128 and currently generates its own quantized test data.
- The CUDA benchmark compares against cuBLAS BF16, not a tensor-wise FP8 cuBLASLt mode; blockwise accuracy and speed must not be inferred from the baseline.
- The SM90 source has not been compiled or executed locally. GPU correctness remains pending even though the CPU algorithm is tested.

## References

- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. arXiv:2412.19437 (fine-grained FP8 scaling).
- Micikevicius et al. *FP8 Formats for Deep Learning.* 2022. arXiv:2209.05433.
- NVIDIA CUTLASS, example 67: Hopper FP8 warp-specialized GEMM with blockwise scaling. https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
- NVIDIA CUTLASS 3.x GEMM API. https://docs.nvidia.com/cutlass/

## License

MIT
