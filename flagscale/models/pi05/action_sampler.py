# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Action Sampling - ODE Sampling for Flow Matching

Reference: openpi/src/openpi/models/pi0.py:217-280

This module implements ODE-based sampling for flow matching models.
Starting from noise, iteratively denoise to get clean actions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

from flagscale.models.pi05.pi05_model import Pi05Model, make_attn_mask


class ActionSampler(nn.Module):
    """
    Action Sampler for ODE-based flow matching sampling.

    Reference: openpi/src/openpi/models/pi0.py:217-280

    Process:
    1. Encode prefix (vision + language) and cache K/V
    2. Iteratively denoise actions using ODE solver
    3. Each step updates actions: x_t += dt * v_t
    """

    def __init__(
        self,
        model: Pi05Model,
        num_steps: int = 10,
        dt: Optional[float] = None
    ):
        """
        Args:
            model: Pi05Model for action prediction
            num_steps: Number of denoising steps
            dt: Time step size (defaults to -1.0/num_steps)
        """
        super().__init__()

        self.model = model
        self.num_steps = num_steps
        self.dt = dt if dt is not None else -1.0 / num_steps

    @torch.no_grad()
    def sample(
        self,
        image_tokens: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample actions using ODE-based flow matching sampling.

        Reference: openpi/src/openpi/models/pi0.py:217-280

        Args:
            image_tokens: [B, T_img, D] pre-computed image tokens
            language_tokens: [B, T_lang] language token IDs (optional)
            image_mask: [B, T_img] image mask (optional)
            language_mask: [B, T_lang] language mask (optional)
            noise: [B, action_horizon, action_dim] initial noise (optional)
            num_steps: Number of sampling steps (optional, overrides default)

        Returns:
            [B, action_horizon, action_dim] sampled actions
        """
        if num_steps is None:
            num_steps = self.num_steps
        dt = self.dt if num_steps == self.num_steps else -1.0 / num_steps

        batch_size = image_tokens.shape[0]
        action_dim = self.model.action_dim
        action_horizon = self.model.action_horizon

        # Initialize noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32, device=image_tokens.device)

        # Initialize state (x_t, time)
        x_t = noise
        time = torch.ones(batch_size, device=x_t.device)  # Start at t=1

        # Iterative denoising loop
        # OpenPI line 239-278
        for step in range(num_steps):
            # Embed prefix (vision + language)
            prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(
                image_tokens=image_tokens,
                language_tokens=language_tokens,
                image_mask=image_mask,
                language_mask=language_mask
            )

            # Embed suffix (current actions with time conditioning)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.model.embed_suffix(
                noisy_actions=x_t,
                timestep=time
            )

            # Combine masks for full sequence
            # OpenPI line 205-207
            input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

            # ar_mask是[B, T]的2D tensor
            # 需要在dim=1 (序列维度) concat
            prefix_ar_mask_1d = prefix_ar_mask[0] if prefix_ar_mask.dim() == 2 else prefix_ar_mask
            suffix_ar_mask_1d = suffix_ar_mask[0] if suffix_ar_mask.dim() == 2 else suffix_ar_mask
            ar_mask_1d = torch.cat([prefix_ar_mask_1d, suffix_ar_mask_1d], dim=0)
            ar_mask = ar_mask_1d.unsqueeze(0).expand_as(input_mask)

            attn_mask = make_attn_mask(input_mask, ar_mask)

            # Compute positions
            # OpenPI line 208
            positions = torch.cumsum(input_mask.int(), dim=1) - 1

            # Forward pass through Gemma
            # OpenPI line 209-211
            (prefix_out, suffix_out), _ = self.model.gemma(
                embedded=[prefix_tokens, suffix_tokens],
                positions=positions,
                attn_mask=attn_mask,
                adarms_cond=[None, adarms_cond]
            )

            # Predict velocity and update actions
            # OpenPI line 212, 271
            suffix_out_for_proj = suffix_out[:, -action_horizon:]

            # Handle dtype conversion
            weight_dtype = self.model.action_out_proj.weight.dtype
            if suffix_out_for_proj.dtype != weight_dtype:
                suffix_out_for_proj = suffix_out_for_proj.to(weight_dtype)

            v_t = self.model.action_out_proj(suffix_out_for_proj)

            # Update x_t (keep as float32 for precision)
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t


def test_action_sampling():
    """测试Action Sampling"""

    print("=" * 80)
    print("ActionSampler 单元测试")
    print("=" * 80)

    batch_size = 2
    num_heads = 4
    num_kv_heads = 1
    head_dim = 32
    num_layers = 2
    vocab_size = 256000
    paligemma_width = 128
    action_expert_width = 256
    ffn_dim = 512
    action_dim = 7
    action_horizon = 5
    num_steps = 5  # Small for testing

    # 测试1: Sampler初始化
    print("\n【测试1】Sampler初始化")
    print("-" * 80)

    model = Pi05Model(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        paligemma_width=paligemma_width,
        action_expert_width=action_expert_width,
        ffn_dim=ffn_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        dtype=torch.bfloat16
    )

    sampler = ActionSampler(model, num_steps=num_steps)

    print(f"  ✅ Sampler初始化成功")
    print(f"  - 采样步数: {num_steps}")
    print(f"  - dt: {sampler.dt:.4f}")

    # 测试2: 基础sampling
    print("\n【测试2】基础Action Sampling")
    print("-" * 80)

    image_tokens = torch.randn(batch_size, 10, paligemma_width, dtype=torch.bfloat16)
    language_tokens = torch.randint(0, vocab_size, (batch_size, 8))

    sampled_actions = sampler.sample(
        image_tokens=image_tokens,
        language_tokens=language_tokens
    )

    assert sampled_actions.shape == (batch_size, action_horizon, action_dim)
    print(f"  ✅ Sampled actions: {sampled_actions.shape}")
    print(f"  ✅ Action range: min={sampled_actions.min():.4f}, max={sampled_actions.max():.4f}")

    # 测试3: 使用自定义noise
    print("\n【测试3】使用自定义Noise")
    print("-" * 80)

    custom_noise = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32) * 0.5
    sampled_actions_custom = sampler.sample(
        image_tokens=image_tokens,
        language_tokens=language_tokens,
        noise=custom_noise
    )

    assert sampled_actions_custom.shape == (batch_size, action_horizon, action_dim)
    print(f"  ✅ Custom noise sampling成功")

    # 测试4: 不同步数测试
    print("\n【测试4】不同采样步数")
    print("-" * 80)

    for steps in [1, 5, 10]:
        actions = sampler.sample(
            image_tokens=image_tokens,
            language_tokens=language_tokens,
            num_steps=steps
        )
        assert actions.shape == (batch_size, action_horizon, action_dim)
        print(f"  ✅ {steps}步采样: shape={actions.shape}, mean={actions.mean():.4f}")

    # 测试5: 确定性（相同noise应该得到相同结果）
    print("\n【测试5】确定性测试")
    print("-" * 80)

    fixed_noise = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)

    actions1 = sampler.sample(
        image_tokens=image_tokens,
        language_tokens=language_tokens,
        noise=fixed_noise
    )

    actions2 = sampler.sample(
        image_tokens=image_tokens,
        language_tokens=language_tokens,
        noise=fixed_noise
    )

    assert torch.allclose(actions1, actions2, atol=1e-6)
    print(f"  ✅ 确定性测试通过：相同noise产生相同结果")

    # 测试6: 不带language tokens
    print("\n【测试6】不带Language Tokens")
    print("-" * 80)

    actions_no_lang = sampler.sample(
        image_tokens=image_tokens,
        language_tokens=None
    )

    assert actions_no_lang.shape == (batch_size, action_horizon, action_dim)
    print(f"  ✅ 无language tokens采样成功")

    print("\n" + "=" * 80)
    print("✅ ActionSampler 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_action_sampling()
