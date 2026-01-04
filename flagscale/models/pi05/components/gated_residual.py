# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Gated Residual Connection - 100% matching OpenPI implementation.

Reference: openpi/src/openpi/models/gemma.py:453-459
"""

import torch
from typing import Optional


def gated_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Gated residual connection.

    精确匹配OpenPI实现: gemma.py:453-459

    三种模式:
    1. gate is None: 标准residual → return x + y
    2. gate is not None: gated residual → return x + y * gate

    关键特性:
    - gate通常来自AdaRMSNorm，shape为[B, 1, D]
    - gate会broadcast到y的shape [B, T, D]
    - 这是AdaRMSNorm的gate机制的核心应用

    Args:
        x: [B, T, D] residual connection的输入 (branch 1)
        y: [B, T, D] residual connection的输入 (branch 2, 通常被gate调制)
        gate: [B, 1, D] or [B, T, D] gate tensor (optional)

    Returns:
        [B, T, D] output = x + (gate * y) or x + y

    实现细节 (逐行对应OpenPI):
    OpenPI代码:
    ```python
    def _gated_residual(x, y, gate):
      if gate is None:
        return x + y
      return x + y * gate
    ```

    使用场景:
    1. Pre-attention norm (无gate):
       residual = gated_residual(x, norm(x), gate=None)

    2. Post-attention (有gate):
       residual = gated_residual(x, attn_output, gate=gate)

    3. Post-FFN (有gate):
       residual = gated_residual(x, ffn_output, gate=gate)
    """
    if gate is None:
        # === 模式1: 标准residual ===
        # OpenPI: return x + y
        return x + y
    else:
        # === 模式2: gated residual ===
        # OpenPI: return x + y * gate
        # gate通常shape为[B, 1, D]，会broadcast到[B, T, D]
        return x + y * gate


def test_gated_residual():
    """验证gated_residual实现的正确性"""

    print("=" * 80)
    print("GatedResidual 单元测试")
    print("=" * 80)

    batch_size, seq_len, dim = 2, 10, 256

    # 测试1: 无gate - 标准residual
    print("\n【测试1】无gate - 应等价于标准residual")
    print("-" * 80)

    x = torch.randn(batch_size, seq_len, dim)
    y = torch.randn(batch_size, seq_len, dim)

    output = gated_residual(x, y, gate=None)
    expected = x + y

    assert torch.allclose(output, expected, atol=1e-6), "无gate时应等于x+y"
    assert output.shape == x.shape, f"Shape错误: {output.shape}"
    print(f"  ✅ 等价于x + y: diff = {(output - expected).abs().max().item():.2e}")
    print(f"  ✅ shape正确: {output.shape}")

    # 测试2: 有gate - gated residual
    print("\n【测试2】有gate - 应等于x + y * gate")
    print("-" * 80)

    x = torch.randn(batch_size, seq_len, dim)
    y = torch.randn(batch_size, seq_len, dim)
    gate = torch.randn(batch_size, 1, dim)  # [B, 1, D] 会broadcast

    output = gated_residual(x, y, gate=gate)
    # gate会broadcast到[B, T, D]
    expected = x + y * gate

    assert torch.allclose(output, expected, atol=1e-6), "有gate时应等于x+y*gate"
    assert output.shape == x.shape, f"Shape错误: {output.shape}"
    print(f"  ✅ 等价于x + y * gate: diff = {(output - expected).abs().max().item():.2e}")
    print(f"  ✅ shape正确: {output.shape}")
    print(f"  ✅ gate broadcast: {gate.shape} → {y.shape}")

    # 测试3: gate shape [B, T, D]
    print("\n【测试3】gate shape为[B, T, D]")
    print("-" * 80)

    x = torch.randn(batch_size, seq_len, dim)
    y = torch.randn(batch_size, seq_len, dim)
    gate = torch.randn(batch_size, seq_len, dim)  # [B, T, D]

    output = gated_residual(x, y, gate=gate)
    expected = x + y * gate

    assert torch.allclose(output, expected, atol=1e-6), "应等于x+y*gate"
    print(f"  ✅ 等价于x + y * gate: diff = {(output - expected).abs().max().item():.2e}")

    # 测试4: gate调制效果
    print("\n【测试4】gate调制效果验证")
    print("-" * 80)

    x = torch.randn(1, 1, dim)
    y = torch.randn(1, 1, dim)

    # gate=1时，应等于标准residual
    gate_ones = torch.ones(1, 1, dim)
    output_ones = gated_residual(x, y, gate=gate_ones)
    output_no_gate = gated_residual(x, y, gate=None)

    assert torch.allclose(output_ones, output_no_gate, atol=1e-6), \
        "gate=1时应等于无gate"
    print(f"  ✅ gate=1时等价于无gate: diff = {(output_ones - output_no_gate).abs().max().item():.2e}")

    # gate=0时，应等于x (y被完全mask)
    gate_zeros = torch.zeros(1, 1, dim)
    output_zeros = gated_residual(x, y, gate=gate_zeros)

    assert torch.allclose(output_zeros, x, atol=1e-6), \
        "gate=0时应等于x"
    print(f"  ✅ gate=0时y被mask: diff with x = {(output_zeros - x).abs().max().item():.2e}")

    # gate=-1时，应等于x - y
    gate_neg = torch.ones(1, 1, dim) * -1
    output_neg = gated_residual(x, y, gate=gate_neg)
    expected_neg = x - y

    assert torch.allclose(output_neg, expected_neg, atol=1e-6), \
        "gate=-1时应等于x-y"
    print(f"  ✅ gate=-1时变成x - y: diff = {(output_neg - expected_neg).abs().max().item():.2e}")

    # 测试5: 梯度传播
    print("\n【测试5】梯度传播测试")
    print("-" * 80)

    x_grad = torch.randn(1, 1, dim, requires_grad=True)
    y_grad = torch.randn(1, 1, dim, requires_grad=True)
    gate_grad = torch.randn(1, 1, dim, requires_grad=True)

    output_grad = gated_residual(x_grad, y_grad, gate=gate_grad)
    loss = output_grad.sum()
    loss.backward()

    assert x_grad.grad is not None, "x梯度为None"
    assert y_grad.grad is not None, "y梯度为None"
    assert gate_grad.grad is not None, "gate梯度为None"
    assert not torch.isnan(x_grad.grad).any(), "x梯度包含NaN"
    assert not torch.isnan(y_grad.grad).any(), "y梯度包含NaN"
    assert not torch.isnan(gate_grad.grad).any(), "gate梯度包含NaN"

    print(f"  ✅ x梯度正常: max = {x_grad.grad.abs().max().item():.2e}")
    print(f"  ✅ y梯度正常: max = {y_grad.grad.abs().max().item():.2e}")
    print(f"  ✅ gate梯度正常: max = {gate_grad.grad.abs().max().item():.2e}")

    # 测试6: dtype保持
    print("\n【测试6】Dtype保持测试")
    print("-" * 80)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_test = torch.randn(2, 5, dim, dtype=dtype)
        y_test = torch.randn(2, 5, dim, dtype=dtype)
        gate_test = torch.randn(2, 1, dim, dtype=dtype)

        output_test = gated_residual(x_test, y_test, gate=gate_test)

        assert output_test.dtype == dtype, f"output dtype {dtype} 未保持"
        print(f"  ✅ {dtype}: OK")

    # 测试7: 与AdaRMSNorm联合测试
    print("\n【测试7】与AdaRMSNorm联合测试")
    print("-" * 80)

    import sys
    sys.path.insert(0, '/nfs/wzp/libero/FlagScale')
    from flagscale.models.pi05.components.adarms_norm import AdaRMSNorm

    dim = 256
    cond_dim = 128

    # 创建AdaRMSNorm
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x = torch.randn(2, 10, dim, dtype=torch.bfloat16)
    cond = torch.randn(2, cond_dim, dtype=torch.float32)

    # 模拟Transformer block中的使用
    # 1. Pre-attention norm (无gate)
    normed1, gate1 = adarms(x, cond=None)
    residual1 = gated_residual(x, normed1, gate=None)

    # 2. Post-attention (有gate)
    normed2, gate2 = adarms(residual1, cond)
    attn_output = torch.randn_like(residual1)  # 模拟attention输出
    residual2 = gated_residual(residual1, attn_output, gate=gate2)

    assert residual1.shape == x.shape, "residual1 shape错误"
    assert residual2.shape == x.shape, "residual2 shape错误"
    assert gate1 is not None, "gate1应为全1tensor"
    assert gate2 is not None, "gate2不应为None"
    assert gate2.shape == (2, 1, dim), f"gate2 shape错误: {gate2.shape}"

    print(f"  ✅ Pre-attention residual正确: {residual1.shape}")
    print(f"  ✅ Post-attention residual正确: {residual2.shape}")
    print(f"  ✅ gate1正确: {gate1.shape}, 全1={torch.allclose(gate1, torch.ones_like(gate1))}")
    print(f"  ✅ gate2正确: {gate2.shape}")

    # 测试8: 边界情况
    print("\n【测试8】边界情况测试")
    print("-" * 80)

    # 零tensor
    x_zero = torch.zeros(1, 1, dim)
    y_zero = torch.zeros(1, 1, dim)
    gate_zero = torch.zeros(1, 1, dim)

    output = gated_residual(x_zero, y_zero, gate=gate_zero)
    assert torch.allclose(output, torch.zeros(1, 1, dim)), "零tensor处理错误"
    print(f"  ✅ 零tensor处理正确")

    # 极大值
    x_large = torch.ones(1, 1, dim) * 1e6
    y_large = torch.ones(1, 1, dim) * 1e6
    gate_small = torch.ones(1, 1, dim) * 1e-6

    output = gated_residual(x_large, y_large, gate=gate_small)
    # 应该约等于 x_large (因为gate很小，y_large的贡献被mask掉)
    expected = x_large + y_large * gate_small

    assert torch.allclose(output, expected, rtol=1e-5), "极大值处理错误"
    print(f"  ✅ 极大值处理正确")

    print("\n" + "=" * 80)
    print("✅ GatedResidual 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_gated_residual()
