# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
AdaRMSNorm Layer - 100% matching OpenPI implementation.

Reference: openpi/src/openpi/models/gemma.py:112-131
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class AdaRMSNorm(nn.Module):
    """
    Adaptive RMS Normalization with conditioning.

    精确匹配OpenPI实现: gemma.py:112-131

    关键特性:
    1. 在float32中计算variance (提高精度)
    2. 支持条件输入 (conditioning)
    3. 从cond预测scale, shift, gate三个调制信号
    4. 返回(output, gate) tuple用于gated residual

    三种模式:
    1. 无AdaRMS层 (dense=None, cond=None): 标准RMSNorm
    2. 有AdaRMS但无cond (dense!=None, cond=None): gate=None
    3. 有AdaRMS有cond (dense!=None, cond!=None): 完整AdaRMS
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        """
        Args:
            dim: hidden dimension
            eps: epsilon for numerical stability (OpenPI uses 1e-6)
            cond_dim: condition dimension (optional). 如果提供，则启用AdaRMS
        """
        super().__init__()
        self.eps = eps
        self.dim = dim

        # 基础scale (用于无cond时的fallback)
        # OpenPI: scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        self.weight = nn.Parameter(torch.zeros(dim))

        # AdaRMS的条件层
        if cond_dim is not None:
            # Dense层: [B, cond_dim] → [B, dim * 3]
            # OpenPI: modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # OpenPI使用zeros_init
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
        else:
            self.dense = None

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: [B, T, D] input tensor
            cond: [B, cond_dim] condition tensor (optional)

        Returns:
            如果dense is None: (output [B, T, D], None)
            如果cond is None: (output [B, T, D], gate None)
            如果cond is not None: (output [B, T, D], gate [B, 1, D])

        实现细节 (逐行对应OpenPI):
        1. 保存原始dtype: dtype = x.dtype
        2. float32中计算var: var = jnp.mean(jnp.square(x.astype(jnp.float32)))
        3. RMS归一化: normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        4. 如果cond is None: 标准RMSNorm
           - scale = self.param("scale", zeros_init)
           - output = normed * (1 + scale)
           - return output.astype(dtype), None
        5. 如果cond is not None: AdaRMS
           - modulation = nn.Dense(cond) → [B, dim*3]
           - scale, shift, gate = jnp.split(modulation, 3, axis=-1)
           - output = normed * (1 + scale) + shift
           - return output.astype(dtype), gate
        """
        dtype = x.dtype
        B, T, D = x.shape

        # Step 1: 在float32中计算variance
        # OpenPI: var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)

        # Step 2: RMS归一化
        # OpenPI: normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        # 关键: 使用原始x，不是x.float()
        x_norm = x * torch.rsqrt(var + self.eps)

        # Step 3: 根据是否有AdaRMS层和condition来处理
        if self.dense is None:
            # === 模式1: 无AdaRMS层 - 标准RMSNorm ===
            # OpenPI: scale = self.param("scale", zeros_init)
            #         normed_inputs = normed_inputs * (1 + scale)
            #         return normed_inputs.astype(dtype), None
            output = x_norm * (1 + self.weight)
            return output.to(dtype), None

        if cond is None:
            # === 模式2: 有AdaRMS层但无condition ===
            # OpenPI: cond=None时不提供gate
            gate = None
            output = x_norm * (1 + self.weight)
            return output.to(dtype), gate

        # === 模式3: 有AdaRMS有condition - 完整AdaRMS ===
        # Step 3a: Dense投影
        # OpenPI: modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros)(cond)
        #          输入: cond [B, cond_dim]
        #          输出: modulation [B, dim * 3]

        # Handle dtype for cond
        weight_dtype = self.dense.weight.dtype
        cond_for_linear = cond.to(weight_dtype) if cond.dtype != weight_dtype else cond
        modulation = self.dense(cond_for_linear)  # [B, dim * 3]

        # Step 3b: 分割成scale, shift, gate
        # OpenPI: scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        #          modulation[:, None, :] 增加一个维度: [B, 1, dim*3]
        #          split后每个变成 [B, 1, dim]
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)  # 每个 [B, D]
        scale = scale.unsqueeze(1)  # [B, 1, D]
        shift = shift.unsqueeze(1)  # [B, 1, D]
        gate = gate.unsqueeze(1)    # [B, 1, D]

        # Step 3c: 应用modulation
        # OpenPI: normed_inputs = normed_inputs * (1 + scale) + shift
        output = x_norm * (1 + scale) + shift

        # Step 3d: 转回原始dtype并返回
        # OpenPI: return normed_inputs.astype(dtype), gate
        return output.to(dtype), gate.to(dtype)


def test_adarms_norm():
    """验证AdaRMSNorm实现的正确性"""
    import sys
    sys.path.insert(0, '/nfs/wzp/libero/FlagScale')
    from flagscale.models.pi05.components.rms_norm import RMSNorm

    print("=" * 80)
    print("AdaRMSNorm 单元测试")
    print("=" * 80)

    batch_size, seq_len, dim = 2, 10, 256
    cond_dim = 128

    # 测试1: 无AdaRMS层 (dense=None)
    print("\n【测试1】无AdaRMS层 - 应等价于RMSNorm")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=None)  # 无AdaRMS
    rms = RMSNorm(dim)

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    output_adarms, gate = adarms(x)
    output_rms = rms(x)

    assert gate is None, "无AdaRMS时gate应为None"
    assert torch.allclose(output_adarms, output_rms, atol=1e-6), "应与RMSNorm一致"
    print(f"  ✅ 与RMSNorm一致: diff = {(output_adarms - output_rms).abs().max().item():.2e}")

    # 测试2: 有AdaRMS层但无cond
    print("\n【测试2】有AdaRMS层但无cond - gate应为None")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    output, gate = adarms(x, cond=None)

    assert gate is None, "cond=None时gate应为None"
    print("  ✅ gate为None")

    # 测试3: 有AdaRMS有cond - 完整功能
    print("\n【测试3】有AdaRMS有cond - 完整功能测试")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    cond = torch.randn(batch_size, cond_dim, dtype=torch.float32)

    output, gate = adarms(x, cond)

    # 检查shape
    assert output.shape == x.shape, f"output shape错误"
    # gate应该是[B, 1, D]，这是正确的
    assert gate.shape[0] == x.shape[0] and gate.shape[2] == x.shape[2], \
        f"gate shape错误: {gate.shape}, 应在第0和2维匹配"
    print(f"  ✅ output shape: {output.shape}")
    print(f"  ✅ gate shape: {gate.shape} (会广播到{seq_len}维)")

    # 检查dtype
    assert output.dtype == x.dtype, f"output dtype错误"
    assert gate.dtype == x.dtype, f"gate dtype错误"
    print(f"  ✅ dtype保持: {output.dtype}")

    # 测试4: AdaRMS的调制效果
    print("\n【测试4】AdaRMS调制效果验证")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x = torch.randn(1, 1, dim, dtype=torch.float32)  # 用float32便于验证
    cond_zero = torch.zeros(1, cond_dim, dtype=torch.float32)
    cond_rand = torch.randn(1, cond_dim, dtype=torch.float32)

    # cond=0时，scale/shift应为0，gate应为0
    output_zero, gate_zero = adarms(x, cond_zero)
    # 由于dense的weight初始化为0，modulation应为0
    # scale=0, shift=0, gate=0
    # 所以 output ≈ x_norm (但会应用weight)
    # gate ≈ 0 (但unsqueeze后是[0], 不为1)
    print(f"  cond=0时:")
    print(f"    gate均值: {gate_zero.mean().item():.6f} (应接近0)")
    print(f"    gate范围: [{gate_zero.min().item():.6f}, {gate_zero.max().item():.6f}]")

    # cond非零时，应有调制效果
    output_rand, gate_rand = adarms(x, cond_rand)
    print(f"  cond≠0时:")
    print(f"    gate均值: {gate_rand.mean().item():.6f}")
    print(f"    gate范围: [{gate_rand.min().item():.6f}, {gate_rand.max().item():.6f}]")
    print(f"    output diff: {(output_rand - output_zero).abs().max().item():.6f}")

    # 测试5: 不同dtype
    print("\n【测试5】Dtype保持测试")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x_test = torch.randn(2, 5, dim, dtype=dtype)
        cond_test = torch.randn(2, cond_dim, dtype=torch.float32)
        output_test, gate_test = adarms(x_test, cond_test)

        assert output_test.dtype == dtype, f"output dtype {dtype} 未保持"
        assert gate_test.dtype == dtype, f"gate dtype {dtype} 未保持"
        print(f"  ✅ {dtype}: OK")

    # 测试6: 梯度传播
    print("\n【测试6】梯度传播测试")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x_grad = torch.randn(1, 1, dim, dtype=torch.float32, requires_grad=True)
    cond_grad = torch.randn(1, cond_dim, dtype=torch.float32, requires_grad=True)

    output_grad, _ = adarms(x_grad, cond_grad)
    loss = output_grad.sum()
    loss.backward()

    assert x_grad.grad is not None, "x梯度为None"
    assert cond_grad.grad is not None, "cond梯度为None"
    assert not torch.isnan(x_grad.grad).any(), "x梯度包含NaN"
    assert not torch.isnan(cond_grad.grad).any(), "cond梯度包含NaN"
    assert not torch.isinf(x_grad.grad).any(), "x梯度包含Inf"
    assert not torch.isinf(cond_grad.grad).any(), "cond梯度包含Inf"

    grad_x_max = x_grad.grad.abs().max()
    grad_cond_max = cond_grad.grad.abs().max()
    print(f"  ✅ x梯度正常: max = {grad_x_max.item():.2e}")
    print(f"  ✅ cond梯度正常: max = {grad_cond_max.item():.2e}")

    # 测试7: 与手动计算对比
    print("\n【测试7】数值正确性验证")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    x = torch.randn(1, 1, dim, dtype=torch.float32)
    cond = torch.randn(1, cond_dim, dtype=torch.float32)

    # 手动计算
    var = torch.mean(torch.square(x), dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + 1e-6)
    modulation = adarms.dense(cond)  # [1, dim*3]
    scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
    scale, shift, gate = scale.unsqueeze(1), shift.unsqueeze(1), gate.unsqueeze(1)
    output_expected = x_norm * (1 + scale) + shift
    gate_expected = gate

    output_actual, gate_actual = adarms(x, cond)

    output_diff = (output_actual - output_expected).abs().max()
    # gate应该完全相同
    gate_diff = (gate_actual - gate_expected).abs().max()

    assert output_diff < 1e-6, f"output数值不匹配: diff = {output_diff}"
    assert gate_diff < 1e-6, f"gate数值不匹配: diff = {gate_diff}"
    print(f"  ✅ output数值正确: diff = {output_diff.item():.2e}")
    print(f"  ✅ gate数值正确: diff = {gate_diff.item():.2e}")

    # 测试8: 权重初始化
    print("\n【测试8】权重初始化测试")
    print("-" * 80)
    adarms = AdaRMSNorm(dim, cond_dim=cond_dim)

    assert torch.allclose(adarms.weight, torch.zeros(dim)), "weight应为zeros_init"
    assert torch.allclose(adarms.dense.weight, torch.zeros(dim * 3, cond_dim)), \
        "dense.weight应为zeros_init"
    print(f"  ✅ weight初始化为zeros")
    print(f"  ✅ dense.weight初始化为zeros")

    print("\n" + "=" * 80)
    print("✅ AdaRMSNorm 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_adarms_norm()
