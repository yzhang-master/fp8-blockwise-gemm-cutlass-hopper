import pytest
import torch

from fp8_blockwise import blockwise_gemm, quantize_blockwise
from fp8_blockwise.quantization import BlockScaledTensor


def _normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()


def test_gemm_matches_explicit_dequantized_product():
    generator = torch.Generator().manual_seed(11)
    a = torch.randn(128, 256, generator=generator)
    b = torch.randn(256, 128, generator=generator)
    qa = quantize_blockwise(a)
    qb = quantize_blockwise(b)

    actual = blockwise_gemm(qa, qb)
    expected = qa.dequantize().float() @ qb.dequantize().float()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)


def test_gemm_is_close_to_fp32_for_dynamic_ranges():
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(130, 256, generator=generator) * 3
    b = torch.randn(256, 129, generator=generator) * 0.1
    actual = blockwise_gemm(quantize_blockwise(a), quantize_blockwise(b))
    expected = a @ b
    assert _normalized_rmse(actual, expected) < 0.06


def test_k_block_scales_affect_their_own_accumulation():
    a = quantize_blockwise(torch.ones(128, 256))
    b = quantize_blockwise(torch.ones(256, 128))
    expected = blockwise_gemm(a, b)
    mutated_scales = a.scales.clone()
    mutated_scales[:, 1] *= 3
    mutated = BlockScaledTensor(a.values, mutated_scales, a.original_shape)
    actual = blockwise_gemm(mutated, b)
    torch.testing.assert_close(actual, expected * 2)


def test_gemm_checks_shapes():
    a = quantize_blockwise(torch.ones(128, 128))
    b = quantize_blockwise(torch.ones(129, 128))
    with pytest.raises(ValueError, match="incompatible GEMM"):
        blockwise_gemm(a, b)
