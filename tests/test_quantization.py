import pytest
import torch

from fp8_blockwise import FP8_E4M3_MAX, quantize_blockwise
from fp8_blockwise.quantization import BlockScaledTensor


def test_quantizes_each_128_square_block_independently():
    tensor = torch.zeros(256, 256)
    tensor[:128, :128] = 1
    tensor[:128, 128:] = 448
    tensor[128:, :128] = -896
    tensor[128:, 128:] = 0.25

    quantized = quantize_blockwise(tensor)

    assert quantized.values.dtype == torch.float8_e4m3fn
    assert quantized.scales.shape == (2, 2)
    torch.testing.assert_close(
        quantized.scales,
        torch.tensor([[1 / 448, 1.0], [2.0, 0.25 / 448]]),
    )
    assert quantized.values.abs().float().max() <= FP8_E4M3_MAX


def test_zero_blocks_use_unit_scale_and_remain_zero():
    quantized = quantize_blockwise(torch.zeros(128, 128))
    assert torch.all(quantized.scales == 1)
    assert torch.count_nonzero(quantized.dequantize()) == 0


def test_dequantization_is_close_for_non_multiple_shape():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(131, 257, generator=generator) * torch.logspace(-2, 2, 257)
    quantized = quantize_blockwise(source)
    restored = quantized.dequantize()
    assert restored.shape == source.shape
    relative_rmse = (restored - source).square().mean().sqrt() / source.square().mean().sqrt()
    assert relative_rmse < 0.08


def test_scale_is_applied_during_dequantization():
    quantized = quantize_blockwise(torch.full((128, 128), 8.0))
    mutated = BlockScaledTensor(
        quantized.values, quantized.scales * 2, quantized.original_shape
    )
    torch.testing.assert_close(mutated.dequantize(), quantized.dequantize() * 2)


def test_rejects_invalid_input():
    with pytest.raises(ValueError, match="rank-2"):
        quantize_blockwise(torch.ones(2, 2, 2))
    with pytest.raises(ValueError, match="NaN"):
        quantize_blockwise(torch.tensor([[float("nan")]]))
    with pytest.raises(TypeError, match="floating-point"):
        quantize_blockwise(torch.ones(2, 2, dtype=torch.int64))
