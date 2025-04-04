from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP8_E4M3_MAX = 448.0
BLOCK_M = 128
BLOCK_K = 128


@dataclass(frozen=True)
class BlockScaledTensor:
    values: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]

    @property
    def padded_shape(self) -> tuple[int, int]:
        return tuple(self.values.shape)  # type: ignore[return-value]

    def dequantize(self) -> torch.Tensor:
        blocks_m = self.padded_shape[0] // BLOCK_M
        blocks_k = self.padded_shape[1] // BLOCK_K
        values = self.values.float().view(
            blocks_m, BLOCK_M, blocks_k, BLOCK_K
        )
        scaled = values * self.scales.float()[:, None, :, None]
        output = scaled.reshape(self.padded_shape)
        return output[: self.original_shape[0], : self.original_shape[1]]


def _ceil_to_block(value: int) -> int:
    return (value + BLOCK_M - 1) // BLOCK_M * BLOCK_M


def quantize_blockwise(tensor: torch.Tensor) -> BlockScaledTensor:
    if tensor.ndim != 2:
        raise ValueError("blockwise quantization expects a rank-2 tensor")
    if not tensor.is_floating_point():
        raise TypeError("blockwise quantization expects a floating-point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("input contains NaN or infinity")

    source = tensor.float()
    rows, cols = source.shape
    padded_rows = _ceil_to_block(rows)
    padded_cols = _ceil_to_block(cols)
    padded = F.pad(source, (0, padded_cols - cols, 0, padded_rows - rows))
    blocks = padded.view(
        padded_rows // BLOCK_M, BLOCK_M, padded_cols // BLOCK_K, BLOCK_K
    )
    absmax = blocks.abs().amax(dim=(1, 3))
    scales = absmax / FP8_E4M3_MAX
    scales = torch.where(absmax == 0, torch.ones_like(scales), scales)
    normalized = blocks / scales[:, None, :, None]
    values = normalized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return BlockScaledTensor(
        values=values.reshape(padded_rows, padded_cols),
        scales=scales,
        original_shape=(rows, cols),
    )
