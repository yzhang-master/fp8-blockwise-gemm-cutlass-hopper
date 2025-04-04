import torch

from .quantization import BLOCK_K, BLOCK_M, BlockScaledTensor


def blockwise_gemm(a: BlockScaledTensor, b: BlockScaledTensor) -> torch.Tensor:
    if a.original_shape[1] != b.original_shape[0]:
        raise ValueError(
            f"incompatible GEMM shapes: {a.original_shape} and {b.original_shape}"
        )
    if a.values.device != b.values.device or a.scales.device != b.scales.device:
        raise ValueError("both operands must be on the same device")
    if a.padded_shape[1] != b.padded_shape[0]:
        raise ValueError("operands have incompatible padded K dimensions")

    a_blocks = a.values.float().view(-1, BLOCK_M, a.padded_shape[1] // BLOCK_K, BLOCK_K)
    b_blocks = b.values.float().view(-1, BLOCK_M, b.padded_shape[1] // BLOCK_K, BLOCK_K)
    output = torch.zeros(
        (a.padded_shape[0], b.padded_shape[1]),
        dtype=torch.float32,
        device=a.values.device,
    )
    for block_k in range(a.padded_shape[1] // BLOCK_K):
        a_values = a_blocks[:, :, block_k, :].reshape(a.padded_shape[0], BLOCK_K)
        b_values = b_blocks[block_k, :, :, :].reshape(BLOCK_K, b.padded_shape[1])
        partial = a_values @ b_values
        scale = a.scales[:, block_k].repeat_interleave(BLOCK_M)[:, None]
        scale = scale * b.scales[block_k, :].repeat_interleave(BLOCK_M)[None, :]
        output.add_(partial * scale)
    return output[: a.original_shape[0], : b.original_shape[1]]
