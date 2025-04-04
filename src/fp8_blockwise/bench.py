import argparse
import json
import platform
from pathlib import Path

import torch

from .gemm import blockwise_gemm
from .quantization import quantize_blockwise


def measure_smoke(seed: int = 123) -> dict[str, float | int | str]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(130, 256, generator=generator)
    b = torch.randn(256, 130, generator=generator)
    qa = quantize_blockwise(a)
    qb = quantize_blockwise(b)
    actual = blockwise_gemm(qa, qb)
    expected = a @ b
    error = actual - expected
    return {
        "device": "cpu",
        "seed": seed,
        "m": a.shape[0],
        "n": b.shape[1],
        "k": a.shape[1],
        "normalized_rmse": float(error.square().mean().sqrt() / expected.square().mean().sqrt()),
        "max_absolute_error": float(error.abs().max()),
        "value_storage_bytes": qa.values.numel() + qb.values.numel(),
        "scale_storage_bytes": (qa.scales.numel() + qb.scales.numel()) * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU correctness smoke; not a GPU performance run")
    parser.add_argument("--smoke", action="store_true", help="run the deterministic CPU case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("use --smoke; Hopper performance is produced by scripts/bench_sm90.sh")
    result = measure_smoke()
    result["torch"] = torch.__version__
    result["python"] = platform.python_version()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
