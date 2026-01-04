# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Multi-Expert Attention Layer - 正确版本

关键理解:
- 每个expert处理不同数量/类型的tokens (不同seq_len)
- 所有expert共享相同的attention配置 (num_heads, num_kv_heads, head_dim)
- QKV在seq_len维度concat，而不是head维度
- 输出按expert切分 (按seq_len切分)

Reference: openpi/src/openpi/models/gemma.py:158-250
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

from flagscale.models.pi05.components.rope import RoPE


class MultiExpertAttention(nn.Module):
    """
    Multi-Expert Attention layer - 正确实现。

    架构:
    - 每个expert处理不同的tokens (不同的T_i)
    - 所有expert共享num_heads, num_kv_heads, head_dim
    - QKV在seq_len维度concat: [B, T0, ...] + [B, T1, ...] -> [B, T0+T1, ...]
    - 联合attention计算
    - 输出按seq_len切分回每个expert
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_experts: int,
        expert_dims: List[int],
        max_seq_len: int = 8192
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.expert_dims = expert_dims
        self.is_gqa = (num_kv_heads != num_heads)

        # 为每个expert创建独立的QKV投影
        if self.is_gqa:
            self.q_projs = nn.ModuleList()
            self.kv_projs = nn.ModuleList()
            for expert_dim in expert_dims:
                self.q_projs.append(nn.Linear(expert_dim, num_heads * head_dim, bias=False))
                self.kv_projs.append(nn.Linear(expert_dim, 2 * num_kv_heads * head_dim, bias=False))
                self._init_lecun_normal(self.q_projs[-1])
                self._init_lecun_normal(self.kv_projs[-1], batch_axis=(0, 1))
        else:
            self.qkv_projs = nn.ModuleList()
            for expert_dim in expert_dims:
                self.qkv_projs.append(nn.Linear(expert_dim, 3 * num_heads * head_dim, bias=False))
                self._init_lecun_normal(self.qkv_projs[-1])

        # RoPE
        self.rope = RoPE(head_dim=head_dim, base=10000.0, max_seq_len=max_seq_len)

        # 输出投影
        self.out_projs = nn.ModuleList()
        for expert_dim in expert_dims:
            self.out_projs.append(nn.Linear(num_heads * head_dim, expert_dim, bias=False))
            self._init_lecun_normal(self.out_projs[-1], in_axis=(-3, -2))

    def _init_lecun_normal(self, layer, **kwargs):
        fan_in = layer.in_features
        std = math.sqrt(1.0 / fan_in)
        nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)

    def forward(
        self,
        xs: List[Optional[torch.Tensor]],
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            xs: list of expert inputs, 每个 [B, T_i, D_i] (不同T_i！)
            positions: [sum(T_i)] position indices
            attn_mask: [B, 1, sum(T_i), sum(T_i)]
            kv_cache: (cached_k, cached_v)
        """
        dtype = next(x.dtype for x in xs if x is not None)
        device = next(x.device for x in xs if x is not None)

        # Step 1: 为每个expert计算QKV
        q_list, k_list, v_list = [], [], []
        seq_lens = []

        if self.is_gqa:
            for i, x in enumerate(xs):
                if x is None:
                    continue
                seq_lens.append(x.shape[1])

                x_dtype = x.dtype
                weight_dtype = self.q_projs[i].weight.dtype
                x_for_linear = x.to(weight_dtype) if x.dtype != weight_dtype else x

                q = self.q_projs[i](x_for_linear)  # [B, T_i, num_heads * head_dim]
                q = q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
                q = q.permute(0, 2, 1, 3)  # [B, num_heads, T_i, head_dim]

                kv = self.kv_projs[i](x_for_linear)  # [B, T_i, 2 * num_kv_heads * head_dim]
                kv = kv.reshape(x.shape[0], x.shape[1], 2, self.num_kv_heads, self.head_dim)
                # kv: [B, T_i, 2, num_kv_heads, head_dim]
                # 分离k和v
                k = kv[:, :, 0].permute(0, 2, 1, 3)  # [B, num_kv_heads, T_i, head_dim]
                v = kv[:, :, 1].permute(0, 2, 1, 3)  # [B, num_kv_heads, T_i, head_dim]

                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
        else:
            for i, x in enumerate(xs):
                if x is None:
                    continue
                seq_lens.append(x.shape[1])

                x_dtype = x.dtype
                weight_dtype = self.qkv_projs[i].weight.dtype
                x_for_linear = x.to(weight_dtype) if x.dtype != weight_dtype else x

                qkv = self.qkv_projs[i](x_for_linear)
                qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
                qkv = qkv.permute(0, 3, 2, 4, 1)  # [B, num_heads, 3, T_i, head_dim]

                q_list.append(qkv[:, 0])
                k_list.append(qkv[:, 1])
                v_list.append(qkv[:, 2])

        # Step 2: 在seq_len维度concat
        # OpenPI line 201: jnp.concatenate(y, axis=1) - axis=1是T维度
        q = torch.cat(q_list, dim=2)  # [B, num_heads, sum(T_i), head_dim]
        k = torch.cat(k_list, dim=2)  # [B, num_kv_heads, sum(T_i), head_dim]
        v = torch.cat(v_list, dim=2)  # [B, num_kv_heads, sum(T_i), head_dim]

        # Step 3: 应用RoPE
        total_seq_len = q.shape[2]
        if positions is None:
            positions = torch.arange(total_seq_len, device=device, dtype=torch.float32)

        # 计算cos/sin
        if total_seq_len <= self.rope.max_seq_len_cached:
            freqs = self.rope.cached_freqs[:total_seq_len]
        else:
            freqs = torch.outer(positions, self.rope.inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]  # [1, 1, total_seq_len, head_dim]
        sin = emb.sin()[None, None, :, :]

        q = self.rope._rotate_with_cos_sin(q, cos, sin)
        k = self.rope._rotate_with_cos_sin(k, cos, sin)

        # Step 4: Scaling
        q = q * (self.head_dim ** -0.5)

        # Step 5: KV cache
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        # Step 6: GQA分组
        num_groups = self.num_heads // self.num_kv_heads
        q = einops.rearrange(q, "B (K G) T H -> B K G T H", K=self.num_kv_heads)
        # q: [B, num_kv_heads, num_groups, T, head_dim]

        # Step 7: Attention计算
        # q: [B, num_kv_heads, num_groups, S, head_dim] (S = total_seq_len)
        # k: [B, num_kv_heads, S, head_dim]
        # GQA attention: 每个kv_head服务num_groups个query heads

        # Step 7: Attention计算
        # q: [B, num_kv_heads, num_groups, S, head_dim] (S = total_seq_len)
        # k: [B, num_kv_heads, S, head_dim]
        # GQA attention: 每个kv_head服务num_groups个query heads
        logits = torch.einsum("BKGSH,BKTH->BKGS T", q.to(torch.float32), k.to(torch.float32))
        # logits: [B, num_kv_heads, num_groups, S, T] where T = S (self-attention)

        # Step 8: Attention mask
        if attn_mask is not None:
            big_neg = -2.3819763e38
            # attn_mask: [B, 1, S, T]
            # logits: [B, num_kv_heads, num_groups, S, T]
            # 需要扩展attn_mask到[B, num_kv_heads, num_groups, S, T]
            attn_mask_2d = attn_mask[:, 0]  # [B, S, T]
            attn_mask_expanded = attn_mask_2d[:, None, None, :, :]  # [B, 1, 1, S, T]
            masked_logits = torch.where(attn_mask_expanded, logits, big_neg)
        else:
            masked_logits = logits

        # Step 9: Softmax (在T维度)
        probs = F.softmax(masked_logits, dim=-1).to(dtype)

        # Step 10: 应用到V
        # v: [B, num_kv_heads, T, head_dim]
        # probs: [B, num_kv_heads, num_groups, S, T]
        encoded = torch.einsum("BKGST,BKTH->BKGSH", probs, v.to(dtype))
        # encoded: [B, num_kv_heads, num_groups, S, head_dim]

        # Step 11: 合并groups
        # encoded: [B, num_kv_heads, num_groups, S, head_dim]
        encoded = einops.rearrange(encoded, "B K G S H -> B S (K G) H")
        # encoded: [B, S, num_heads, head_dim]

        # Step 12: 为每个expert切分输出
        outputs = []
        start = 0
        for i, x in enumerate(xs):
            if x is None:
                outputs.append(None)
                continue

            end = start + seq_lens[i]
            expert_encoded = encoded[:, start:end]  # [B, T_i, num_heads, head_dim]

            # 输出投影
            expert_encoded_flat = expert_encoded.reshape(expert_encoded.shape[0], expert_encoded.shape[1], -1)

            x_dtype = x.dtype
            weight_dtype = self.out_projs[i].weight.dtype
            if expert_encoded_flat.dtype != weight_dtype:
                expert_encoded_flat = expert_encoded_flat.to(weight_dtype)

            expert_output = self.out_projs[i](expert_encoded_flat).to(x_dtype)
            outputs.append(expert_output)
            start = end

        return outputs, (k, v)


def test_multi_expert_attention_v2():
    """测试正确的Multi-Expert Attention实现"""

    print("=" * 80)
    print("MultiExpertAttention V2 单元测试")
    print("=" * 80)

    batch_size = 2
    num_heads = 4
    num_kv_heads = 1  # 极端GQA，类似OpenPI
    head_dim = 32
    expert_dims = [128, 256]
    seq_lens = [10, 5]  # 不同expert有不同seq_len！

    attn = MultiExpertAttention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=2,
        expert_dims=expert_dims
    )

    xs = [
        torch.randn(batch_size, seq_lens[0], expert_dims[0], dtype=torch.bfloat16),
        torch.randn(batch_size, seq_lens[1], expert_dims[1], dtype=torch.bfloat16)
    ]

    total_seq_len = sum(seq_lens)
    attn_mask = torch.ones(batch_size, 1, total_seq_len, total_seq_len, dtype=torch.bool)

    outputs, (k, v) = attn(xs, attn_mask=attn_mask)

    assert len(outputs) == 2
    assert outputs[0].shape == xs[0].shape
    assert outputs[1].shape == xs[1].shape

    print(f"  ✅ Expert 0 output: {outputs[0].shape}")
    print(f"  ✅ Expert 1 output: {outputs[1].shape}")
    print(f"  ✅ K shape: {k.shape}, V shape: {v.shape}")
    print("\n" + "=" * 80)
    print("✅ MultiExpertAttention V2 测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_multi_expert_attention_v2()
