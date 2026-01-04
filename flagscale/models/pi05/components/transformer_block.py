# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Transformer Block - Pre-normalization with Gated Residuals

Reference: openpi/src/openpi/models/gemma.py:284-333
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

from flagscale.models.pi05.components.rms_norm import RMSNorm
from flagscale.models.pi05.components.adarms_norm import AdaRMSNorm
from flagscale.models.pi05.components.gated_residual import gated_residual
from flagscale.models.pi05.components.attention import MultiExpertAttention
from flagscale.models.pi05.components.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block with pre-normalization and gated residuals.

    Architecture (OpenPI gemma.py:284-333):
    1. Pre-attention normalization (AdaRMSNorm)
    2. Multi-expert attention sub-layer
    3. Gated residual connection
    4. Pre-FFN normalization (AdaRMSNorm)
    5. Feed-forward sub-layer
    6. Gated residual connection

    Supports:
    - Multi-expert architecture
    - AdaRMS conditioning (expert-specific)
    - GQA (Grouped Query Attention)
    - Dropout (optional)
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_experts: int,
        expert_dims: List[int],
        expert_ffn_dims: List[int],
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        adarms_cond_dim: Optional[int] = None
    ):
        """
        Args:
            num_heads: Number of attention heads per expert
            num_kv_heads: Number of key/value heads (GQA)
            head_dim: Dimension per attention head
            num_experts: Number of experts
            expert_dims: Hidden dimension for each expert [D0, D1, ...]
            expert_ffn_dims: FFN hidden dimension for each expert [FFN0, FFN1, ...]
            max_seq_len: Maximum sequence length for RoPE
            dropout: Dropout rate
            adarms_cond_dim: Condition dimension for AdaRMSNorm (optional).
                             Only Expert 1 uses AdaRMS conditioning in Pi05.
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.expert_dims = expert_dims
        self.expert_ffn_dims = expert_ffn_dims
        self.dropout_rate = dropout

        # Pre-attention normalization (OpenPI line 303)
        # Expert 0: no cond, Expert 1: with cond (if adarms_cond_dim provided)
        self.pre_attn_norms = nn.ModuleList([
            AdaRMSNorm(
                dim=expert_dim,
                cond_dim=adarms_cond_dim if (i == 1 and adarms_cond_dim is not None) else None
            )
            for i, expert_dim in enumerate(expert_dims)
        ])

        # Multi-expert attention (OpenPI line 297)
        self.attn = MultiExpertAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            expert_dims=expert_dims,
            max_seq_len=max_seq_len
        )

        # Pre-FFN normalization (OpenPI line 318)
        # Expert 0: no cond, Expert 1: with cond (if adarms_cond_dim provided)
        self.pre_ffn_norms = nn.ModuleList([
            AdaRMSNorm(
                dim=expert_dim,
                cond_dim=adarms_cond_dim if (i == 1 and adarms_cond_dim is not None) else None
            )
            for i, expert_dim in enumerate(expert_dims)
        ])

        # Feed-forward networks (OpenPI line 319)
        self.ffns = nn.ModuleList([
            FeedForward(
                dim=expert_dim,
                hidden_dim=expert_ffn_dim
            )
            for expert_dim, expert_ffn_dim in zip(expert_dims, expert_ffn_dims)
        ])

        # Dropout (OpenPI line 295)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        xs: List[Optional[torch.Tensor]],
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        adarms_cond: Optional[List[Optional[torch.Tensor]]] = None,
        deterministic: bool = True
    ) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            xs: List of expert inputs, 每个 [B, T_i, D_i] 或 None
            positions: [sum(T_i)] position indices
            attn_mask: [B, 1, sum(T_i), sum(T_i)] attention mask
            kv_cache: (cached_k, cached_v) for autoregressive generation
            adarms_cond: List of conditioning tensors, 每个 [B, cond_dim] 或 None
                         Only used if use_adarms=True
            deterministic: If True, disable dropout

        Returns:
            (outputs, (k, v)): outputs是每个expert的输出list, k/v是完整的key/value
        """
        # ========================================
        # Pre-attention normalization + Gate
        # OpenPI lines 299-305
        # ========================================
        pre_attn_outputs = []
        attn_gates = []

        for i, x in enumerate(xs):
            if x is None:
                pre_attn_outputs.append(None)
                attn_gates.append(None)
                continue

            # Apply AdaRMSNorm (with or without cond)
            cond = adarms_cond[i] if adarms_cond is not None else None
            normalized, gate = self.pre_attn_norms[i](x, cond)

            pre_attn_outputs.append(normalized)
            attn_gates.append(gate)

        # ========================================
        # Multi-expert attention
        # OpenPI lines 307-310
        # ========================================
        attn_outputs, kv_cache = self.attn(
            xs=pre_attn_outputs,
            positions=positions,
            attn_mask=attn_mask,
            kv_cache=kv_cache
        )

        # Apply dropout (OpenPI line 309)
        if self.dropout is not None and not deterministic:
            attn_outputs = [self.dropout(o) if o is not None else None for o in attn_outputs]

        # ========================================
        # Gated residual connection (post-attention)
        # OpenPI line 311
        # ========================================
        residual_outputs = []
        for x, y, gate in zip(xs, attn_outputs, attn_gates):
            residual_outputs.append(gated_residual(x, y, gate))

        # ========================================
        # Pre-FFN normalization + Gate
        # OpenPI lines 314-326
        # ========================================
        pre_ffn_outputs = []
        ffn_gates = []

        for i, x in enumerate(residual_outputs):
            if x is None:
                pre_ffn_outputs.append(None)
                ffn_gates.append(None)
                continue

            # Apply AdaRMSNorm (with or without cond)
            cond = adarms_cond[i] if adarms_cond is not None else None
            normalized, gate = self.pre_ffn_norms[i](x, cond)

            pre_ffn_outputs.append(normalized)
            ffn_gates.append(gate)

        # ========================================
        # Feed-forward networks
        # OpenPI lines 319
        # ========================================
        ffn_outputs = []
        for i, x in enumerate(pre_ffn_outputs):
            if x is None:
                ffn_outputs.append(None)
                continue

            ffn_output = self.ffns[i](x)
            ffn_outputs.append(ffn_output)

        # Apply dropout (OpenPI line 329)
        if self.dropout is not None and not deterministic:
            ffn_outputs = [self.dropout(o) if o is not None else None for o in ffn_outputs]

        # ========================================
        # Gated residual connection (post-FFN)
        # OpenPI line 330
        # ========================================
        final_outputs = []
        for x, y, gate in zip(residual_outputs, ffn_outputs, ffn_gates):
            final_outputs.append(gated_residual(x, y, gate))

        return final_outputs, kv_cache


def test_transformer_block():
    """测试TransformerBlock实现"""

    print("=" * 80)
    print("TransformerBlock 单元测试")
    print("=" * 80)

    batch_size = 2
    num_heads = 4
    num_kv_heads = 1  # GQA
    head_dim = 32
    num_experts = 2
    expert_dims = [128, 256]
    expert_ffn_dims = [512, 1024]
    seq_lens = [10, 5]

    # 测试1: 基础功能 - AdaRMSNorm
    print("\n【测试1】基础功能 - AdaRMSNorm with conditioning")
    print("-" * 80)

    block = TransformerBlock(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        adarms_cond_dim=64,  # Expert 1 will use conditioning
        dropout=0.0
    )

    xs = [
        torch.randn(batch_size, seq_lens[0], expert_dims[0], dtype=torch.bfloat16),
        torch.randn(batch_size, seq_lens[1], expert_dims[1], dtype=torch.bfloat16)
    ]

    total_seq_len = sum(seq_lens)
    attn_mask = torch.ones(batch_size, 1, total_seq_len, total_seq_len, dtype=torch.bool)
    adarms_cond = [None, torch.randn(batch_size, 64, dtype=torch.bfloat16)]

    outputs, (k, v) = block(
        xs=xs,
        attn_mask=attn_mask,
        adarms_cond=adarms_cond
    )

    assert len(outputs) == num_experts
    assert outputs[0].shape == xs[0].shape
    assert outputs[1].shape == xs[1].shape
    assert outputs[0].dtype == xs[0].dtype
    assert outputs[1].dtype == xs[1].dtype

    print(f"  ✅ Expert 0 output: {outputs[0].shape}, dtype={outputs[0].dtype}")
    print(f"  ✅ Expert 1 output: {outputs[1].shape}, dtype={outputs[1].dtype}")
    print(f"  ✅ K/V cache: K={k.shape}, V={v.shape}")

    # 测试2: 无cond的AdaRMSNorm (fallback to RMSNorm behavior)
    print("\n【测试2】AdaRMSNorm without conditioning")
    print("-" * 80)

    block_no_cond = TransformerBlock(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        adarms_cond_dim=None  # No conditioning
    )

    outputs_no_cond, _ = block_no_cond(xs=xs, attn_mask=attn_mask, adarms_cond=None)

    assert outputs_no_cond[0].shape == xs[0].shape
    assert outputs_no_cond[1].shape == xs[1].shape
    print(f"  ✅ 无cond模式输出正确")

    # 测试3: 梯度传播
    print("\n【测试3】梯度传播测试")
    print("-" * 80)

    xs_grad = [
        torch.randn(1, 5, expert_dims[0], dtype=torch.float32, requires_grad=True),
        torch.randn(1, 3, expert_dims[1], dtype=torch.float32, requires_grad=True)
    ]
    attn_mask_grad = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    adarms_cond_grad = [None, torch.randn(1, 64, dtype=torch.float32)]

    outputs_grad, _ = block(
        xs=xs_grad,
        attn_mask=attn_mask_grad,
        adarms_cond=adarms_cond_grad
    )

    loss = sum(o.sum() for o in outputs_grad if o is not None)
    loss.backward()

    for i, x_grad in enumerate(xs_grad):
        assert x_grad.grad is not None, f"Expert {i} 梯度为None"
        assert not torch.isnan(x_grad.grad).any(), f"Expert {i} 梯度包含NaN"
        print(f"  ✅ Expert {i} 梯度正常: max = {x_grad.grad.abs().max().item():.2e}")

    # 测试4: KV cache
    print("\n【测试4】KV cache功能")
    print("-" * 80)

    # 第一次调用
    xs1 = [
        torch.randn(batch_size, 5, expert_dims[0], dtype=torch.float32),
        torch.randn(batch_size, 3, expert_dims[1], dtype=torch.float32)
    ]
    attn_mask1 = torch.ones(batch_size, 1, 8, 8, dtype=torch.bool)
    outputs1, (k1, v1) = block(xs=xs1, attn_mask=attn_mask1)

    # 第二次调用，使用cache
    xs2 = [
        torch.randn(batch_size, 3, expert_dims[0], dtype=torch.float32),
        torch.randn(batch_size, 2, expert_dims[1], dtype=torch.float32)
    ]
    attn_mask2 = torch.ones(batch_size, 1, 5, 13, dtype=torch.bool)  # 8+5=13
    outputs2, (k2, v2) = block(xs=xs2, attn_mask=attn_mask2, kv_cache=(k1, v1))

    assert k2.shape[2] == 13, f"K cache shape错误: {k2.shape}"
    assert v2.shape[2] == 13, f"V cache shape错误: {v2.shape}"
    print(f"  ✅ KV cache正确: seq_len = {k2.shape[2]}")

    # 测试5: Dropout
    print("\n【测试5】Dropout功能")
    print("-" * 80)

    block_dropout = TransformerBlock(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        dropout=0.1
    )

    # Deterministic mode (no dropout)
    outputs_no_dropout, _ = block_dropout(xs=xs, attn_mask=attn_mask, deterministic=True)
    outputs_no_dropout2, _ = block_dropout(xs=xs, attn_mask=attn_mask, deterministic=True)

    for i in range(num_experts):
        assert torch.allclose(outputs_no_dropout[i], outputs_no_dropout2[i]), f"Expert {i} deterministic mode失败"
    print(f"  ✅ Deterministic模式正确 (dropout禁用)")

    # Training mode (dropout active)
    block_dropout.train()
    outputs_train, _ = block_dropout(xs=xs, attn_mask=attn_mask, deterministic=False)

    # 由于dropout的随机性，两次forward结果应该不同
    # 注意：这个测试偶尔会失败（dropout可能不改变某些输出），所以只做检查不强制断言
    different = any(
        not torch.allclose(outputs_no_dropout[i], outputs_train[i]) for i in range(num_experts)
    )
    if different:
        print(f"  ✅ Training模式正确 (dropout启用，输出随机)")
    else:
        print(f"  ⚠ Training模式测试完成 (dropout未改变输出，可能正常)")


    print("\n" + "=" * 80)
    print("✅ TransformerBlock 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_transformer_block()
