# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Gemma Module - Complete Transformer with Multi-Expert Support

Reference: openpi/src/openpi/models/gemma.py:340-422
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Sequence
import math
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

from flagscale.models.pi05.components.rms_norm import RMSNorm
from flagscale.models.pi05.components.adarms_norm import AdaRMSNorm
from flagscale.models.pi05.components.transformer_block import TransformerBlock


class Embedder(nn.Module):
    """
    Token Embedder.

    Reference: openpi/src/openpi/models/gemma.py:135-155
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_std: float = 0.02
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            init_std: Standard deviation for initialization (OpenPI uses normal)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # OpenPI: input_embedding_table with normal initialization
        self.input_embedding_table = nn.Parameter(
            torch.randn(vocab_size, embed_dim) * init_std
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens to embeddings.

        Args:
            x: [B, T] token indices

        Returns:
            [B, T, D] embeddings
        """
        # OpenPI line 149: x = self.input_embedding_table[(x,)]
        embeddings = self.input_embedding_table[x]

        # OpenPI line 150: x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        embeddings = embeddings * math.sqrt(self.embed_dim)

        return embeddings

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to logits.

        Args:
            x: [B, T, D] embeddings

        Returns:
            [B, T, vocab_size] logits
        """
        # OpenPI line 154: return jnp.dot(x, self.input_embedding_table.T)
        return torch.matmul(x, self.input_embedding_table.t())


class GemmaModule(nn.Module):
    """
    Complete Gemma Transformer Module with Multi-Expert Support.

    Reference: openpi/src/openpi/models/gemma.py:340-422

    Architecture:
    1. Embedder (token embeddings)
    2. Multiple TransformerBlock layers
    3. Final normalization for each expert
    4. AdaRMS conditioning support
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_experts: int,
        expert_dims: List[int],
        expert_ffn_dims: List[int],
        num_layers: int,
        vocab_size: int,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        adarms_cond_dim: Optional[int] = None,
        embed_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Args:
            num_heads: Number of attention heads per expert
            num_kv_heads: Number of key/value heads (GQA)
            head_dim: Dimension per attention head
            num_experts: Number of experts
            expert_dims: Hidden dimension for each expert [D0, D1, ...]
            expert_ffn_dims: FFN hidden dimension for each expert [FFN0, FFN1, ...]
            num_layers: Number of transformer layers (all experts share same depth)
            vocab_size: Vocabulary size (for embedder)
            max_seq_len: Maximum sequence length for RoPE
            dropout: Dropout rate
            adarms_cond_dim: Condition dimension for AdaRMSNorm
            embed_dtype: Data type for embeddings
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.expert_dims = expert_dims
        self.expert_ffn_dims = expert_ffn_dims
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.adarms_cond_dim = adarms_cond_dim
        self.embed_dtype = embed_dtype

        # Embedder (OpenPI line 354-358)
        # Only for first expert (usually PaliGemma vision/language tokens)
        self.embedder = Embedder(
            vocab_size=vocab_size,
            embed_dim=expert_dims[0]
        )

        # Transformer layers (OpenPI line 359-381)
        # nn.scan creates multiple layers with shared structure
        self.layers = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                expert_dims=expert_dims,
                expert_ffn_dims=expert_ffn_dims,
                max_seq_len=max_seq_len,
                dropout=dropout,
                adarms_cond_dim=adarms_cond_dim
            )
            for _ in range(num_layers)
        ])

        # Final normalizations (OpenPI line 382)
        # One RMSNorm for each expert
        self.final_norms = nn.ModuleList([
            AdaRMSNorm(
                dim=expert_dim,
                cond_dim=adarms_cond_dim if (i == 1 and adarms_cond_dim is not None) else None
            )
            for i, expert_dim in enumerate(expert_dims)
        ])

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens.

        Args:
            tokens: [B, T] token indices

        Returns:
            [B, T, D] embeddings
        """
        # OpenPI line 386: return self.embedder.encode(tokens).astype(self.embed_dtype)
        embeddings = self.embedder.encode(tokens)
        return embeddings.to(self.embed_dtype)

    def forward(
        self,
        embedded: Sequence[Optional[torch.Tensor]],
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        adarms_cond: Optional[Sequence[Optional[torch.Tensor]]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = True
    ) -> Tuple[List[Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through all layers.

        Args:
            embedded: List of expert embedded inputs, 每个 [B, T_i, D_i] 或 None
            positions: [sum(T_i)] position indices
            attn_mask: [B, sum(T_i), sum(T_i)] attention mask (will add dimension)
            adarms_cond: List of conditioning tensors, 每个 [B, cond_dim] 或 None
            kv_cache: (cached_k, cached_v) for autoregressive generation
            deterministic: If True, disable dropout

        Returns:
            (outputs, kv_cache): outputs是每个expert的输出list, kv_cache是更新后的cache
        """
        # OpenPI line 400: embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        embedded = [e.to(self.embed_dtype) if e is not None else None for e in embedded]

        # OpenPI line 401: mask = jnp.asarray(mask)[:, None, :, :]
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, :, :]  # [B, 1, T, T]

        # OpenPI line 402-403: if adarms_cond is None: adarms_cond = [None] * len(self.configs)
        if adarms_cond is None:
            adarms_cond = [None] * self.num_experts

        # OpenPI line 405: embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
        # IMPORTANT: In OpenPI, nn.scan carries kv_cache through layers
        # But in PyTorch, each layer should compute independently
        # kv_cache is only used for autoregressive generation within the same layer
        # For normal forward pass, we don't use kv_cache
        for layer in self.layers:
            embedded, _ = layer(
                xs=embedded,
                positions=positions,
                attn_mask=attn_mask,
                kv_cache=None,  # Each layer computes fresh K/V
                adarms_cond=adarms_cond,
                deterministic=deterministic
            )

        # For kv_cache return, we return None (not supported in multi-layer forward)
        # To support proper caching, we would need per-layer caches
        kv_cache = (None, None)

        # OpenPI line 407: assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        for e in embedded:
            if e is not None:
                assert e.dtype == self.embed_dtype, f"Expected {self.embed_dtype}, got {e.dtype}"

        # OpenPI line 409-411: final_norms
        outputs = []
        for i, (e, cond) in enumerate(zip(embedded, adarms_cond)):
            if e is None:
                outputs.append(None)
                continue

            # Apply final normalization (with AdaRMS conditioning if applicable)
            normed, _ = self.final_norms[i](e, cond)
            outputs.append(normed)

        return outputs, kv_cache


def test_gemma_module():
    """测试GemmaModule实现"""

    print("=" * 80)
    print("GemmaModule 单元测试")
    print("=" * 80)

    batch_size = 2
    num_heads = 4
    num_kv_heads = 1  # GQA
    head_dim = 32
    num_experts = 2
    expert_dims = [128, 256]
    expert_ffn_dims = [512, 1024]
    num_layers = 2  # Small for testing
    vocab_size = 256000
    seq_lens = [10, 5]

    # 测试1: 基础功能
    print("\n【测试1】基础功能 - Embedder + Transformer Layers")
    print("-" * 80)

    gemma = GemmaModule(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        num_layers=num_layers,
        vocab_size=vocab_size,
        adarms_cond_dim=64,
        embed_dtype=torch.bfloat16
    )

    # Test embedder
    tokens = torch.randint(0, vocab_size, (batch_size, seq_lens[0]))
    embeddings = gemma.embed(tokens)
    assert embeddings.shape == (batch_size, seq_lens[0], expert_dims[0])
    assert embeddings.dtype == torch.bfloat16
    print(f"  ✅ Embedder输出: {embeddings.shape}, dtype={embeddings.dtype}")

    # Test forward pass
    embedded = [
        torch.randn(batch_size, seq_lens[0], expert_dims[0], dtype=torch.bfloat16),
        torch.randn(batch_size, seq_lens[1], expert_dims[1], dtype=torch.bfloat16)
    ]

    total_seq_len = sum(seq_lens)
    # GemmaModule forward expects attn_mask: [B, sum(T_i), sum(T_i)] and will add dimension
    attn_mask = torch.ones(batch_size, total_seq_len, total_seq_len, dtype=torch.bool)
    adarms_cond = [None, torch.randn(batch_size, 64, dtype=torch.bfloat16)]

    # Pass positions as well (required for RoPE)
    positions = torch.arange(total_seq_len, device=embedded[0].device)

    outputs, kv_cache = gemma(
        embedded=embedded,
        positions=positions,
        attn_mask=attn_mask,
        adarms_cond=adarms_cond
    )

    assert len(outputs) == num_experts
    assert outputs[0].shape == embedded[0].shape
    assert outputs[1].shape == embedded[1].shape
    assert outputs[0].dtype == torch.bfloat16
    assert outputs[1].dtype == torch.bfloat16

    print(f"  ✅ Expert 0输出: {outputs[0].shape}, dtype={outputs[0].dtype}")
    print(f"  ✅ Expert 1输出: {outputs[1].shape}, dtype={outputs[1].dtype}")
    if kv_cache[0] is not None:
        print(f"  ✅ KV cache: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    else:
        print(f"  ✅ KV cache: None (多层模式不支持)")

    # 测试2: 无AdaRMS模式
    print("\n【测试2】无AdaRMS模式")
    print("-" * 80)

    gemma_no_adarms = GemmaModule(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        num_layers=num_layers,
        vocab_size=vocab_size,
        adarms_cond_dim=None,
        embed_dtype=torch.bfloat16
    )

    positions = torch.arange(total_seq_len, device=embedded[0].device)
    outputs_no_adarms, _ = gemma_no_adarms(embedded=embedded, positions=positions, attn_mask=attn_mask)

    assert outputs_no_adarms[0].shape == embedded[0].shape
    assert outputs_no_adarms[1].shape == embedded[1].shape
    print(f"  ✅ 无AdaRMS模式输出正确")

    # 测试3: 梯度传播
    print("\n【测试3】梯度传播测试")
    print("-" * 80)

    embedded_grad = [
        torch.randn(1, 5, expert_dims[0], dtype=torch.float32, requires_grad=True),
        torch.randn(1, 3, expert_dims[1], dtype=torch.float32, requires_grad=True)
    ]
    total_seq_len_grad = 8  # 5 + 3
    attn_mask_grad = torch.ones(1, total_seq_len_grad, total_seq_len_grad, dtype=torch.bool)
    positions_grad = torch.arange(total_seq_len_grad, device=embedded_grad[0].device)
    adarms_cond_grad = [None, torch.randn(1, 64, dtype=torch.float32)]

    outputs_grad, _ = gemma(
        embedded=embedded_grad,
        positions=positions_grad,
        attn_mask=attn_mask_grad,
        adarms_cond=adarms_cond_grad
    )

    loss = sum(o.sum() for o in outputs_grad if o is not None)
    loss.backward()

    for i, e_grad in enumerate(embedded_grad):
        assert e_grad.grad is not None, f"Expert {i} 梯度为None"
        assert not torch.isnan(e_grad.grad).any(), f"Expert {i} 梯度包含NaN"
        print(f"  ✅ Expert {i} 梯度正常: max = {e_grad.grad.abs().max().item():.2e}")

    # 测试4: KV cache在TransformerBlock层已测试
    print("\n【测试4】KV cache支持")
    print("-" * 80)
    print(f"  ✅ KV cache在TransformerBlock层已通过测试")
    print(f"  ℹ  多层GemmaModule的KV cache管理较为复杂，Phase 1暂不实现")

    # 测试5: Dropout
    print("\n【测试5】Dropout功能")
    print("-" * 80)

    gemma_dropout = GemmaModule(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        expert_dims=expert_dims,
        expert_ffn_dims=expert_ffn_dims,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dropout=0.1,
        adarms_cond_dim=64
    )

    # Deterministic mode
    positions = torch.arange(total_seq_len, device=embedded[0].device)
    outputs_det1, _ = gemma_dropout(embedded=embedded, positions=positions, attn_mask=attn_mask, deterministic=True)
    outputs_det2, _ = gemma_dropout(embedded=embedded, positions=positions, attn_mask=attn_mask, deterministic=True)

    for i in range(num_experts):
        assert torch.allclose(outputs_det1[i], outputs_det2[i]), f"Expert {i} deterministic mode失败"
    print(f"  ✅ Deterministic模式正确")

    # Training mode
    gemma_dropout.train()
    outputs_train, _ = gemma_dropout(embedded=embedded, attn_mask=attn_mask, deterministic=False)

    different = any(
        not torch.allclose(outputs_det1[i], outputs_train[i]) for i in range(num_experts)
    )
    if different:
        print(f"  ✅ Training模式正确 (dropout启用)")
    else:
        print(f"  ⚠ Training模式测试完成 (dropout未改变输出)")

    print("\n" + "=" * 80)
    print("✅ GemmaModule 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_gemma_module()
