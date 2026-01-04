# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Pi05 Model - Vision-Language-Action Flow Matching Model

Reference: openpi/src/openpi/models/pi0.py:66-280

Architecture:
1. Vision Encoder (SigLIP) - Encode images to tokens
2. Language Embeddings - Tokenized prompts
3. GemmaModule (Multi-Expert) - Process vision+language (Expert 0) and actions (Expert 1)
4. Action Projections - Map actions to/from embedding space
5. Time Encoding - Encode timestep for AdaRMS conditioning
6. Flow Matching - Train and sample actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import math
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

from flagscale.models.pi05.components.gemma_module import GemmaModule


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0
) -> torch.Tensor:
    """
    Computes sine-cosine positional embedding vectors for scalar positions.

    Reference: openpi/src/openpi/models/pi0.py:48-63

    Args:
        pos: [B] positions in [0, 1]
        embedding_dim: Dimension of embedding
        min_period: Minimum period
        max_period: Maximum period

    Returns:
        [B, embedding_dim] positional embeddings
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # Create frequency range
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute sinusoid input
    # pos: [B], 1/period: [embedding_dim/2]
    sinusoid_input = torch.einsum("i,j->ij", pos, 1.0 / period * 2 * math.pi)

    # Concatenate sin and cos
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


def make_attn_mask(
    input_mask: torch.Tensor,
    mask_ar: torch.Tensor
) -> torch.Tensor:
    """
    Create attention mask from input mask and autoregressive mask.

    Reference: openpi/src/openpi/models/pi0.py:19-44

    Args:
        input_mask: [B, N] true if part of input, false if padding
        mask_ar: [N] or [B, N] true where previous tokens cannot depend on it

    Returns:
        [B, N, N] attention mask

    Note:
        mask_ar can be 1D [N] or 2D [B, N]. If 1D, it will be broadcast to [B, N].
        This matches OpenPI's behavior where mask_ar is created by concat on axis=0.
    """
    # Broadcast mask_ar to match input_mask shape (OpenPI line 40)
    if mask_ar.dim() == 1:
        # [N] -> [B, N] by broadcasting
        mask_ar = mask_ar.unsqueeze(0).expand_as(input_mask)
    elif mask_ar.shape != input_mask.shape:
        # [B, N] -> ensure correct shape
        mask_ar = mask_ar.expand_as(input_mask)

    # Cumulative sum of autoregressive mask (OpenPI line 41)
    cumsum = torch.cumsum(mask_ar.float(), dim=1)

    # Create causal attention mask (OpenPI line 42)
    # Tokens can attend if their cumsum <= current token's cumsum
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

    # Valid mask (both tokens must be valid) (OpenPI line 43)
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    return torch.logical_and(attn_mask, valid_mask)


class Pi05Model(nn.Module):
    """
    Pi05 Model - Vision-Language-Action Flow Matching Model

    Reference: openpi/src/openpi/models/pi0.py:66-280

    Architecture:
    - Vision Encoder: SigLIP for image encoding
    - Language Embeddings: Tokenized prompts
    - GemmaModule: Multi-expert transformer
      * Expert 0: PaliGemma (vision + language)
      * Expert 1: Actions with AdaRMS conditioning
    - Flow Matching: Train and sample actions
    """

    def __init__(
        self,
        # Gemma configuration
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        paligemma_width: int,
        action_expert_width: int,
        ffn_dim: int,
        action_dim: int,
        action_horizon: int,
        action_expert_ffn_dim: Optional[int] = None,
        time_embed_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Args:
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads (GQA)
            head_dim: Dimension per attention head
            num_layers: Number of transformer layers
            vocab_size: Vocabulary size
            paligemma_width: Hidden dimension for PaliGemma expert
            action_expert_width: Hidden dimension for action expert
            ffn_dim: Feed-forward network hidden dimension
            action_expert_ffn_dim: FFN hidden dimension for action expert (defaults to ffn_dim)
            max_seq_len: Maximum sequence length
            action_dim: Action dimension
            action_horizon: Number of action steps to predict
            time_embed_dim: Time embedding dimension (defaults to action_expert_width)
            dropout: Dropout rate
            dtype: Data type for embeddings
        """
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.dtype = dtype

        # Time embedding dimension (for AdaRMS conditioning)
        if time_embed_dim is None:
            time_embed_dim = action_expert_width
        if action_expert_ffn_dim is None:
            action_expert_ffn_dim = ffn_dim

        # GemmaModule with multi-expert configuration
        # Expert 0: PaliGemma (vision + language)
        # Expert 1: Actions (with AdaRMS conditioning)
        self.gemma = GemmaModule(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=2,
            expert_dims=[paligemma_width, action_expert_width],
            expert_ffn_dims=[ffn_dim, action_expert_ffn_dim],
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            adarms_cond_dim=time_embed_dim,  # Expert 1 uses AdaRMS
            embed_dtype=dtype
        )

        # Action input projection: [action_dim] -> [action_expert_width]
        self.action_in_proj = nn.Linear(action_dim, action_expert_width)
        self.action_in_proj.to(dtype=dtype)

        # Time encoding MLP (for AdaRMS conditioning)
        self.time_mlp_in = nn.Linear(time_embed_dim, action_expert_width)
        self.time_mlp_in.to(dtype=dtype)
        self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        self.time_mlp_out.to(dtype=dtype)

        # Action output projection: [action_expert_width] -> [action_dim]
        self.action_out_proj = nn.Linear(action_expert_width, action_dim)
        self.action_out_proj.to(dtype=dtype)

    def embed_time(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Encode timestep using sine-cosine embedding + MLP.

        Reference: openpi/src/openpi/models/pi0.py:161-167

        Args:
            timestep: [B] timestep in [0, 1]

        Returns:
            [B, action_expert_width] time embedding for AdaRMS conditioning
        """
        # Sine-cosine positional encoding
        # OpenPI: min_period=4e-3, max_period=4.0
        time_emb = posemb_sincos(
            timestep,
            self.time_mlp_in.in_features,
            min_period=4e-3,
            max_period=4.0
        )

        # Convert to the dtype of the MLP weights
        weight_dtype = self.time_mlp_in.weight.dtype
        if time_emb.dtype != weight_dtype:
            time_emb = time_emb.to(weight_dtype)

        # MLP to transform for AdaRMS conditioning
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)  # swish = silu
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)

        return time_emb

    def embed_prefix(
        self,
        image_tokens: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed prefix (vision + language tokens).

        Reference: openpi/src/openpi/models/pi0.py:106-137

        Args:
            image_tokens: [B, T_img, D] pre-computed image tokens from SigLIP
            language_tokens: [B, T_lang] language token IDs (optional)
            image_mask: [B, T_img] image mask (optional)
            language_mask: [B, T_lang] language mask (optional)

        Returns:
            tokens: [B, T_prefix, D] prefix tokens
            input_mask: [B, T_prefix] input mask
            ar_mask: [B, T_prefix] autoregressive mask
        """
        tokens_list = []
        input_mask_list = []
        ar_mask_list = []

        # Image tokens
        tokens_list.append(image_tokens)
        if image_mask is not None:
            input_mask_list.append(image_mask)
        else:
            input_mask_list.append(torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device))
        # Image tokens attend to each other (not autoregressive)
        ar_mask_list.extend([False] * image_tokens.shape[1])

        # Language tokens
        if language_tokens is not None:
            # Embed language tokens using Gemma's embedder
            lang_embeds = self.gemma.embed(language_tokens)
            tokens_list.append(lang_embeds)

            if language_mask is not None:
                input_mask_list.append(language_mask)
            else:
                input_mask_list.append(torch.ones(language_tokens.shape[:2], dtype=torch.bool, device=language_tokens.device))

            # Full attention between image and language
            ar_mask_list.extend([False] * language_tokens.shape[1])

        # Concatenate
        tokens = torch.cat(tokens_list, dim=1)
        input_mask = torch.cat(input_mask_list, dim=1)
        ar_mask = torch.tensor(ar_mask_list, dtype=torch.bool, device=tokens.device)  # 1D tensor [total_len]

        return tokens, input_mask, ar_mask

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed suffix (action tokens with time conditioning).

        Reference: openpi/src/openpi/models/pi0.py:140-186

        Args:
            noisy_actions: [B, action_horizon, action_dim] noisy actions
            timestep: [B] timestep in [0, 1]
            state: [B, state_dim] robot state (optional, not used in Pi05)

        Returns:
            tokens: [B, T_suffix, D] suffix tokens
            input_mask: [B, T_suffix] input mask
            ar_mask: [B, T_suffix] autoregressive mask
            adarms_cond: [B, D] time conditioning for AdaRMS
        """
        batch_size = noisy_actions.shape[0]

        # Project actions to embedding space
        # Handle dtype conversion
        weight_dtype = self.action_in_proj.weight.dtype
        noisy_actions_for_proj = noisy_actions.to(weight_dtype) if noisy_actions.dtype != weight_dtype else noisy_actions
        action_tokens = self.action_in_proj(noisy_actions_for_proj)  # [B, action_horizon, action_expert_width]

        # Convert back to original dtype
        if action_tokens.dtype != noisy_actions.dtype:
            action_tokens = action_tokens.to(noisy_actions.dtype)

        # Time embedding for AdaRMS conditioning
        time_emb = self.embed_time(timestep)  # [B, action_expert_width]
        adarms_cond = time_emb

        # In Pi05, we don't mix time into action tokens (unlike Pi0)
        # Time is only used for AdaRMS conditioning
        tokens = action_tokens

        # Create masks
        input_mask = torch.ones(batch_size, tokens.shape[1], dtype=torch.bool, device=tokens.device)

        # Actions are causal: first action blocks all, rest can attend to previous
        # This matches OpenPI's causal attention for actions
        ar_mask = [True] + [False] * (self.action_horizon - 1)
        ar_mask = torch.tensor(ar_mask, dtype=torch.bool, device=tokens.device)  # 1D tensor [action_horizon]

        return tokens, input_mask, ar_mask, adarms_cond

    def forward(
        self,
        image_tokens: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass - compute loss or predict actions.

        Reference: openpi/src/openpi/models/pi0.py:189-214

        Args:
            image_tokens: [B, T_img, D] pre-computed image tokens
            noisy_actions: [B, action_horizon, action_dim] noisy actions at timestep t
            timestep: [B] timestep in [0, 1]
            language_tokens: [B, T_lang] language token IDs (optional)
            image_mask: [B, T_img] image mask (optional)
            language_mask: [B, T_lang] language mask (optional)
            target_actions: [B, action_horizon, action_dim] target actions (for loss computation)
            return_loss: If True, compute loss; otherwise predict actions

        Returns:
            predicted_actions: [B, action_horizon, action_dim]
            loss: scalar loss (if return_loss=True)
        """
        batch_size = image_tokens.shape[0]

        # Embed prefix (vision + language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            image_tokens=image_tokens,
            language_tokens=language_tokens,
            image_mask=image_mask,
            language_mask=language_mask
        )

        # Embed suffix (actions with time conditioning)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            noisy_actions=noisy_actions,
            timestep=timestep
        )

        # Combine masks
        # OpenPI line 206: ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        # Both prefix_ar_mask and suffix_ar_mask are 1D tensors, concat on dim=0
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)  # Concat 1D tensors
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Compute positions
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        # Forward through GemmaModule
        # Expert 0: prefix (vision + language)
        # Expert 1: suffix (actions with AdaRMS)
        (prefix_out, suffix_out), _ = self.gemma(
            embedded=[prefix_tokens, suffix_tokens],
            positions=positions,
            attn_mask=attn_mask,
            adarms_cond=[None, adarms_cond]
        )

        # Project back to action space
        suffix_out_for_proj = suffix_out[:, -self.action_horizon:]

        # Handle dtype conversion for Linear layer
        weight_dtype = self.action_out_proj.weight.dtype
        if suffix_out_for_proj.dtype != weight_dtype:
            suffix_out_for_proj = suffix_out_for_proj.to(weight_dtype)

        predicted_actions = self.action_out_proj(suffix_out_for_proj)

        # Convert back to original dtype
        if predicted_actions.dtype != self.dtype:
            predicted_actions = predicted_actions.to(self.dtype)

        # Compute loss if requested
        loss = None
        if return_loss and target_actions is not None:
            # Flow matching loss: MSE(v_t, u_t)
            # Reference: openpi/src/openpi/models/pi0.py:214
            # OpenPI: return jnp.mean(jnp.square(v_t - u_t), axis=-1)
            # This means mean over action_dim, keeping [B, action_horizon]

            # Ensure target_actions is in the same dtype as predicted_actions
            target_actions_aligned = target_actions
            if target_actions.dtype != predicted_actions.dtype:
                target_actions_aligned = target_actions.to(predicted_actions.dtype)

            # Compute squared error
            squared_error = torch.square(predicted_actions - target_actions_aligned)

            # Mean over action_dim (axis=-1)
            # Result shape: [B, action_horizon]
            loss_per_action_dim = torch.mean(squared_error, dim=-1)

            # Then global mean over batch and horizon
            loss = loss_per_action_dim.mean()

        return predicted_actions, loss

    def sample_actions(
        self,
        image_tokens: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample actions using flow matching inference.

        Reference: openpi/src/openpi/models/pi0.py:216-279

        This method implements the flow matching sampling loop:
        1. Start with pure noise
        2. Iteratively denoise using the learned velocity field
        3. Return the denoised actions

        Args:
            image_tokens: [B, T_img, D] pre-computed image tokens from vision encoder
            language_tokens: [B, T_lang] language token IDs (optional)
            image_mask: [B, T_img] image mask (optional)
            language_mask: [B, T_lang] language mask (optional)
            state: [B, state_dim] robot state (optional)
            num_steps: Number of sampling steps (default: 10)
            rng: Random number generator (optional)

        Returns:
            [B, action_horizon, action_dim] sampled actions
        """
        # Backward-compat for older positional calls:
        # sample_actions(image_tokens, language_tokens, state, num_steps, rng)
        if image_mask is not None and language_mask is not None:
            if isinstance(language_mask, int) and isinstance(state, torch.Generator) and rng is None:
                rng = state
                num_steps = language_mask
                state = image_mask
                image_mask = None
                language_mask = None
        if image_mask is not None and state is None:
            if image_mask.dtype is not torch.bool and image_mask.shape != image_tokens.shape[:2]:
                state = image_mask
                image_mask = None

        if rng is None:
            rng = torch.Generator(device=image_tokens.device)
            rng.seed()

        batch_size = image_tokens.shape[0]

        # Step 1: Embed prefix (vision + language)
        # OpenPI line 234-237
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            image_tokens=image_tokens,
            language_tokens=language_tokens,
            image_mask=image_mask,
            language_mask=language_mask,
        )

        # Create attention mask for prefix
        # Note: prefix_ar_mask is already 1D from embed_prefix
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Compute positions for prefix
        prefix_positions = torch.cumsum(prefix_mask.int(), dim=1) - 1

        # Step 2: Forward pass to fill KV cache with prefix
        # OpenPI line 237
        # Note: Current implementation doesn't support KV cache, so we recompute prefix each time
        # This is less efficient but functionally correct
        # TODO: Implement KV cache for efficiency

        # Step 3: Initialize with noise
        # OpenPI line 230-231
        noise = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            generator=rng,
            dtype=self.dtype,
            device=image_tokens.device
        )

        # Step 4: Flow matching sampling loop
        # OpenPI uses convention: t=1 is noise, t=0 is target
        # Line 227-228
        dt = -1.0 / num_steps  # Time step size

        x_t = noise  # Current state (starts at noise)
        time = torch.ones(batch_size, dtype=self.dtype, device=image_tokens.device)  # Starts at t=1

        # Iterative denoising
        # OpenPI line 239-278
        for step in range(num_steps):
            # Embed suffix (current noisy actions + time)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                noisy_actions=x_t,
                timestep=time,
            )

            # Combine prefix/suffix masks to build full attention mask
            # OpenPI line 246-252
            all_input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
            all_ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
            full_attn_mask = make_attn_mask(all_input_mask, all_ar_mask)

            # Compute positions for suffix
            # OpenPI line 259
            suffix_positions = (
                torch.sum(prefix_mask, dim=-1, keepdim=True) +
                torch.cumsum(suffix_mask.int(), dim=1) - 1
            )

            # Combined positions
            all_positions = torch.cat([prefix_positions, suffix_positions], dim=1)

            # Forward pass
            # OpenPI line 261-268
            (prefix_out, suffix_out), _ = self.gemma(
                embedded=[prefix_tokens, suffix_tokens],
                positions=all_positions,
                attn_mask=full_attn_mask,
                adarms_cond=[None, adarms_cond],
            )

            # Project to action space and get velocity
            # OpenPI line 269
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            # Euler integration step
            # OpenPI line 271
            x_t = x_t + dt * v_t
            time = time + dt

            # Early stopping if time reaches 0
            # OpenPI line 273-276
            if (time < -dt / 2).all():
                break

        # x_t is now the denoised actions (t=0)
        # OpenPI line 278-279
        return x_t


def test_pi05_model():
    """测试Pi05模型实现"""

    print("=" * 80)
    print("Pi05Model 单元测试")
    print("=" * 80)

    batch_size = 2
    num_heads = 4
    num_kv_heads = 1  # GQA
    head_dim = 32
    num_layers = 2
    vocab_size = 256000
    paligemma_width = 128
    action_expert_width = 256
    ffn_dim = 512
    action_dim = 7
    action_horizon = 5

    # 测试1: 模型初始化
    print("\n【测试1】模型初始化")
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

    print(f"  ✅ 模型初始化成功")
    print(f"  - Gemma专家维度: {paligemma_width}, {action_expert_width}")
    print(f"  - Action维度: {action_dim}")
    print(f"  - Action horizon: {action_horizon}")

    # 测试2: Time encoding
    print("\n【测试2】Time Encoding")
    print("-" * 80)

    timestep = torch.rand(batch_size)
    time_emb = model.embed_time(timestep)

    assert time_emb.shape == (batch_size, action_expert_width)
    print(f"  ✅ Time embedding shape: {time_emb.shape}")

    # 测试3: Prefix embedding
    print("\n【测试3】Prefix Embedding")
    print("-" * 80)

    image_tokens = torch.randn(batch_size, 10, paligemma_width, dtype=torch.bfloat16)
    language_tokens = torch.randint(0, vocab_size, (batch_size, 8))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
        image_tokens=image_tokens,
        language_tokens=language_tokens
    )

    assert prefix_tokens.shape[0] == batch_size
    assert prefix_tokens.shape[2] == paligemma_width
    print(f"  ✅ Prefix tokens: {prefix_tokens.shape}")
    print(f"  ✅ Prefix mask: {prefix_mask.shape}")
    print(f"  ✅ Prefix AR mask: {prefix_ar_mask.shape}")

    # 测试4: Suffix embedding
    print("\n【测试4】Suffix Embedding")
    print("-" * 80)

    noisy_actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.bfloat16)
    timestep = torch.rand(batch_size)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        noisy_actions=noisy_actions,
        timestep=timestep
    )

    assert suffix_tokens.shape == (batch_size, action_horizon, action_expert_width)
    assert adarms_cond.shape == (batch_size, action_expert_width)
    print(f"  ✅ Suffix tokens: {suffix_tokens.shape}")
    print(f"  ✅ AdaRMS conditioning: {adarms_cond.shape}")

    # 测试5: Forward pass with loss
    print("\n【测试5】Forward Pass with Loss")
    print("-" * 80)

    target_actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.bfloat16)

    predicted_actions, loss = model(
        image_tokens=image_tokens,
        noisy_actions=noisy_actions,
        timestep=timestep,
        language_tokens=language_tokens,
        target_actions=target_actions,
        return_loss=True
    )

    assert predicted_actions.shape == (batch_size, action_horizon, action_dim)
    assert loss is not None
    print(f"  ✅ Predicted actions: {predicted_actions.shape}")
    print(f"  ✅ Loss: {loss.item():.6f}")

    # 测试6: Gradient flow
    print("\n【测试6】Gradient Flow")
    print("-" * 80)

    # Create new inputs for gradient test
    noisy_actions_grad = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.bfloat16)
    timestep_grad = torch.rand(batch_size)
    target_actions_grad = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.bfloat16)

    predicted_actions_grad, loss_grad = model(
        image_tokens=image_tokens,
        noisy_actions=noisy_actions_grad,
        timestep=timestep_grad,
        language_tokens=language_tokens,
        target_actions=target_actions_grad,
        return_loss=True
    )

    loss_grad.backward()

    # Check gradients exist
    assert model.action_out_proj.weight.grad is not None
    assert not torch.isnan(model.action_out_proj.weight.grad).any()
    print(f"  ✅ 梯度正常: max = {model.action_out_proj.weight.grad.abs().max().item():.2e}")

    print("\n" + "=" * 80)
    print("✅ Pi05Model 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_pi05_model()
