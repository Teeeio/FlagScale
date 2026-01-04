# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
RMSNorm Layer - 100% matching OpenPI implementation.

Reference: openpi/src/openpi/models/gemma.py:112-131
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    精确匹配OpenPI实现: gemma.py:112-131

    关键特性:
    1. 在float32中计算variance (提高精度)
    2. RMS归一化公式: x * rsqrt(var + eps)
    3. 应用可学习的scale: output = x_norm * (1 + weight)
    4. 返回原始dtype
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: hidden dimension
            eps: epsilon for numerical stability (OpenPI uses 1e-6)
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        # OpenPI使用zeros_init (nn.initializers.zeros_init())
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, D] input tensor

        Returns:
            [B, T, D] normalized output, same dtype as input

        实现细节 (逐行对应OpenPI):
        1. 保存原始dtype: dtype = x.dtype
        2. float32中计算var: var = jnp.mean(jnp.square(x.astype(jnp.float32)))
        3. RMS归一化: normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        4. 应用scale: output = normed * (1 + scale)
        5. 转回dtype: return output.astype(dtype)
        """
        dtype = x.dtype

        # Step 1: 在float32中计算variance
        # OpenPI: var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)

        # Step 2: RMS归一化
        # OpenPI: normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        # 关键: 这里用的是原始x，不是x.float()！
        # 但是rsqrt的结果是float32，所以x会自动cast
        x_norm = x * torch.rsqrt(var + self.eps)

        # Step 3: 应用可学习的scale
        # OpenPI: normed_inputs = normed_inputs * (1 + scale)
        output = x_norm * (1 + self.weight)

        # Step 4: 返回原始dtype
        # OpenPI: return normed_inputs.astype(dtype)
        return output.to(dtype)


def test_rms_norm():
    """验证RMSNorm实现的正确性"""
    import math

    print("=" * 80)
    print("RMSNorm 单元测试")
    print("=" * 80)

    # 创建测试数据
    batch_size, seq_len, dim = 2, 10, 256
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    # 测试1: 基础功能
    print("\n【测试1】基础功能测试")
    norm = RMSNorm(dim)
    output = norm(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert output.dtype == x.dtype, f"Dtype mismatch: {output.dtype} != {x.dtype}"
    print(f"  ✅ Shape正确: {output.shape}")
    print(f"  ✅ Dtype保持: {output.dtype}")

    # 测试2: 手动验证数值
    print("\n【测试2】数值正确性验证")
    # 关键：使用原始x（可能是bfloat16），不是x_fp32
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    x_norm_expected = x * torch.rsqrt(var + 1e-6)  # 使用原始x
    output_expected = x_norm_expected * (1 + norm.weight)
    # 然后转回原始dtype
    output_expected = output_expected.to(x.dtype)

    # 现在应该完全匹配
    assert torch.allclose(output, output_expected, atol=1e-6), \
        f"数值不匹配: max diff = {(output - output_expected).abs().max()}"
    print(f"  ✅ 数值正确: max diff = {(output - output_expected).abs().max().item():.2e}")

    # 测试3: 不同dtype
    print("\n【测试3】Dtype保持测试")
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_test = torch.randn(2, 5, dim, dtype=dtype)  # 使用相同的dim
        output_test = norm(x_test)
        assert output_test.dtype == dtype, f"Dtype {dtype} 未保持"
        print(f"  ✅ {dtype}: OK")

    # 测试4: 梯度传播
    print("\n【测试4】梯度传播测试")
    x_grad = torch.randn(1, 1, dim, dtype=torch.float32, requires_grad=True)
    norm_fp32 = RMSNorm(dim)
    output_grad = norm_fp32(x_grad)
    loss = output_grad.sum()
    loss.backward()

    assert x_grad.grad is not None, "梯度为None"
    assert not torch.isnan(x_grad.grad).any(), "梯度包含NaN"
    assert not torch.isinf(x_grad.grad).any(), "梯度包含Inf"
    assert x_grad.grad.abs().max() < 1e3, f"梯度爆炸: {x_grad.grad.abs().max()}"
    print(f"  ✅ 梯度正常: max = {x_grad.grad.abs().max().item():.2e}")

    # 测试5: 权重初始化
    print("\n【测试5】权重初始化测试")
    norm_new = RMSNorm(dim)
    assert torch.allclose(norm_new.weight, torch.zeros(dim)), \
        "权重应为zeros_init"
    print(f"  ✅ 权重初始化为zeros")

    print("\n" + "=" * 80)
    print("✅ RMSNorm 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_rms_norm()
