#!/usr/bin/env python3
"""
Convert OpenPI pi05 JAX checkpoint params to FlagScale Pi05Model weights.

Usage example (run in an env with JAX/Orbax/Flax installed):
  python tools/convert_openpi_pi05_to_flagscale.py \
    --checkpoint_dir /share/pi05_models/openpi05_base \
    --output_path /share/pi05_models/openpi05_base/flagscale_pi05.safetensors \
    --action_horizon 50
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax import traverse_util
from safetensors.torch import save_file

from flagscale.models.pi05.pi05_model import Pi05Model


def restore_params(
    params_path: Union[Path, str],
    *,
    restore_type: Union[type[np.ndarray], type[jax.Array]] = jax.Array,
    dtype: Optional[jnp.dtype] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Dict:
    """Restore params from OpenPI-style Orbax checkpoints without OpenPI dependencies."""
    params_path = Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    flat_params = traverse_util.flatten_dict(params)
    if flat_params and all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def _build_paligemma_config():
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    return PaliGemmaConfig()


def _slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format (ported from OpenPI)."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def _cast_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    if dtype is None:
        return {k: v.contiguous() for k, v in state_dict.items()}
    out = {}
    for key, value in state_dict.items():
        if value.dtype != dtype:
            value = value.to(dtype)
        out[key] = value.contiguous()
    return out


def export_openpi_vlm_weights(params: Dict, output_dir: Path, precision: str) -> None:
    paligemma_params = traverse_util.flatten_dict(params["PaliGemma"], sep="/")
    paligemma_config = _build_paligemma_config()
    paligemma_state, _ = _slice_paligemma_state_dict(paligemma_params, paligemma_config)

    vlm_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    paligemma_state = _cast_state_dict(paligemma_state, vlm_dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(paligemma_state, str(output_dir / "model.safetensors"))


def _to_torch(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.contiguous()


def _transpose_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}")
    return array.T


def _reshape_q(array: np.ndarray) -> np.ndarray:
    # JAX q_einsum: [num_heads, hidden_size, head_dim]
    # PyTorch q_proj: [num_heads * head_dim, hidden_size]
    num_heads, hidden_size, head_dim = array.shape
    return array.transpose(0, 2, 1).reshape(num_heads * head_dim, hidden_size)


def _reshape_o(array: np.ndarray) -> np.ndarray:
    # JAX attn_vec_einsum: [num_heads, head_dim, hidden_size]
    # PyTorch out_proj: [hidden_size, num_heads * head_dim]
    num_heads, head_dim, hidden_size = array.shape
    return array.transpose(2, 0, 1).reshape(hidden_size, num_heads * head_dim)


def _reshape_kv(array: np.ndarray) -> np.ndarray:
    # JAX kv_einsum: [2, num_kv_heads, hidden_size, head_dim]
    # PyTorch kv_proj: [2 * num_kv_heads * head_dim, hidden_size]
    if array.ndim != 4:
        raise ValueError(f"Expected kv array with 4 dims, got {array.shape}")
    key = array[0, 0].T
    value = array[1, 0].T
    return np.concatenate([key, value], axis=0)


def _collect_bias_warnings(flat: Dict[str, np.ndarray], keys: List[str]) -> List[str]:
    warnings = []
    for key in keys:
        if key not in flat:
            continue
        bias = flat[key]
        max_abs = float(np.max(np.abs(bias)))
        if max_abs > 1e-6:
            warnings.append(f"{key} bias max_abs={max_abs:.6e} (no bias in FlagScale)")
    return warnings


def _infer_config(flat: Dict[str, np.ndarray]) -> Dict[str, int]:
    q = flat["PaliGemma/llm/layers/attn/q_einsum/w"]
    q1 = flat["PaliGemma/llm/layers/attn/q_einsum_1/w"]
    kv = flat["PaliGemma/llm/layers/attn/kv_einsum/w"]
    ffn0 = flat["PaliGemma/llm/layers/mlp/gating_einsum"]
    ffn1 = flat["PaliGemma/llm/layers/mlp_1/gating_einsum"]
    embed = flat["PaliGemma/llm/embedder/input_embedding"]
    action_in = flat["action_in_proj/kernel"]

    return {
        "num_layers": q.shape[0],
        "num_heads": q.shape[1],
        "paligemma_width": q.shape[2],
        "head_dim": q.shape[3],
        "action_expert_width": q1.shape[2],
        "num_kv_heads": kv.shape[2],
        "ffn_dim": ffn0.shape[3],
        "action_expert_ffn_dim": ffn1.shape[3],
        "vocab_size": embed.shape[0],
        "action_dim": action_in.shape[0],
    }


def build_state_dict(
    flat: Dict[str, np.ndarray],
    action_horizon: int,
    dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], List[str]]:
    config = _infer_config(flat)
    model = Pi05Model(
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        paligemma_width=config["paligemma_width"],
        action_expert_width=config["action_expert_width"],
        ffn_dim=config["ffn_dim"],
        action_expert_ffn_dim=config["action_expert_ffn_dim"],
        action_dim=config["action_dim"],
        action_horizon=action_horizon,
        dtype=dtype,
    )
    state = model.state_dict()
    mapped = {}

    # Embedding
    mapped["gemma.embedder.input_embedding_table"] = _to_torch(
        flat["PaliGemma/llm/embedder/input_embedding"], dtype
    )

    # Projections
    mapped["action_in_proj.weight"] = _to_torch(
        _transpose_2d(flat["action_in_proj/kernel"]), dtype
    )
    mapped["action_in_proj.bias"] = _to_torch(flat["action_in_proj/bias"], dtype)
    mapped["action_out_proj.weight"] = _to_torch(
        _transpose_2d(flat["action_out_proj/kernel"]), dtype
    )
    mapped["action_out_proj.bias"] = _to_torch(flat["action_out_proj/bias"], dtype)
    mapped["time_mlp_in.weight"] = _to_torch(
        _transpose_2d(flat["time_mlp_in/kernel"]), dtype
    )
    mapped["time_mlp_in.bias"] = _to_torch(flat["time_mlp_in/bias"], dtype)
    mapped["time_mlp_out.weight"] = _to_torch(
        _transpose_2d(flat["time_mlp_out/kernel"]), dtype
    )
    mapped["time_mlp_out.bias"] = _to_torch(flat["time_mlp_out/bias"], dtype)

    # Final norms
    mapped["gemma.final_norms.0.weight"] = _to_torch(
        flat["PaliGemma/llm/final_norm/scale"], dtype
    )
    mapped["gemma.final_norms.1.dense.weight"] = _to_torch(
        _transpose_2d(flat["PaliGemma/llm/final_norm_1/Dense_0/kernel"]), dtype
    )
    mapped["gemma.final_norms.1.dense.bias"] = _to_torch(
        flat["PaliGemma/llm/final_norm_1/Dense_0/bias"], dtype
    )

    # Per-layer mapping
    q = flat["PaliGemma/llm/layers/attn/q_einsum/w"]
    q1 = flat["PaliGemma/llm/layers/attn/q_einsum_1/w"]
    kv = flat["PaliGemma/llm/layers/attn/kv_einsum/w"]
    kv1 = flat["PaliGemma/llm/layers/attn/kv_einsum_1/w"]
    o = flat["PaliGemma/llm/layers/attn/attn_vec_einsum/w"]
    o1 = flat["PaliGemma/llm/layers/attn/attn_vec_einsum_1/w"]
    pre_attn = flat["PaliGemma/llm/layers/pre_attention_norm/scale"]
    pre_ffn = flat["PaliGemma/llm/layers/pre_ffw_norm/scale"]
    pre_attn_1 = flat["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel"]
    pre_ffn_1 = flat["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel"]
    pre_attn_1_bias = flat["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias"]
    pre_ffn_1_bias = flat["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias"]
    ffn = flat["PaliGemma/llm/layers/mlp/gating_einsum"]
    ffn1 = flat["PaliGemma/llm/layers/mlp_1/gating_einsum"]
    ffn_linear = flat["PaliGemma/llm/layers/mlp/linear"]
    ffn_linear1 = flat["PaliGemma/llm/layers/mlp_1/linear"]

    for i in range(config["num_layers"]):
        mapped[f"gemma.layers.{i}.attn.q_projs.0.weight"] = _to_torch(_reshape_q(q[i]), dtype)
        mapped[f"gemma.layers.{i}.attn.q_projs.1.weight"] = _to_torch(_reshape_q(q1[i]), dtype)
        mapped[f"gemma.layers.{i}.attn.kv_projs.0.weight"] = _to_torch(_reshape_kv(kv[i]), dtype)
        mapped[f"gemma.layers.{i}.attn.kv_projs.1.weight"] = _to_torch(_reshape_kv(kv1[i]), dtype)
        mapped[f"gemma.layers.{i}.attn.out_projs.0.weight"] = _to_torch(_reshape_o(o[i]), dtype)
        mapped[f"gemma.layers.{i}.attn.out_projs.1.weight"] = _to_torch(_reshape_o(o1[i]), dtype)

        mapped[f"gemma.layers.{i}.pre_attn_norms.0.weight"] = _to_torch(pre_attn[i], dtype)
        mapped[f"gemma.layers.{i}.pre_attn_norms.1.dense.weight"] = _to_torch(
            _transpose_2d(pre_attn_1[i]), dtype
        )
        mapped[f"gemma.layers.{i}.pre_attn_norms.1.dense.bias"] = _to_torch(
            pre_attn_1_bias[i], dtype
        )
        mapped[f"gemma.layers.{i}.pre_ffn_norms.0.weight"] = _to_torch(pre_ffn[i], dtype)
        mapped[f"gemma.layers.{i}.pre_ffn_norms.1.dense.weight"] = _to_torch(
            _transpose_2d(pre_ffn_1[i]), dtype
        )
        mapped[f"gemma.layers.{i}.pre_ffn_norms.1.dense.bias"] = _to_torch(
            pre_ffn_1_bias[i], dtype
        )

        mapped[f"gemma.layers.{i}.ffns.0.gate.weight"] = _to_torch(
            _transpose_2d(ffn[i, 0]), dtype
        )
        mapped[f"gemma.layers.{i}.ffns.0.up.weight"] = _to_torch(
            _transpose_2d(ffn[i, 1]), dtype
        )
        mapped[f"gemma.layers.{i}.ffns.0.down.weight"] = _to_torch(
            _transpose_2d(ffn_linear[i]), dtype
        )

        mapped[f"gemma.layers.{i}.ffns.1.gate.weight"] = _to_torch(
            _transpose_2d(ffn1[i, 0]), dtype
        )
        mapped[f"gemma.layers.{i}.ffns.1.up.weight"] = _to_torch(
            _transpose_2d(ffn1[i, 1]), dtype
        )
        mapped[f"gemma.layers.{i}.ffns.1.down.weight"] = _to_torch(
            _transpose_2d(ffn_linear1[i]), dtype
        )

    # Fill into state dict for shape validation
    for key, tensor in mapped.items():
        if key not in state:
            raise KeyError(f"Key not in FlagScale model: {key}")
        if state[key].shape != tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: {state[key].shape} vs {tensor.shape}")
        state[key] = tensor

    bias_warnings = []

    return state, config, bias_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenPI pi05 JAX weights to FlagScale Pi05Model.")
    parser.add_argument("--checkpoint_dir", required=True, help="OpenPI checkpoint dir (contains params/).")
    parser.add_argument("--output_path", required=True, help="Output safetensors path.")
    parser.add_argument("--action_horizon", type=int, default=50, help="Action horizon for model init.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Output dtype.")
    parser.add_argument(
        "--export_vlm",
        action="store_true",
        help="Also export OpenPI PyTorch VLM (pi0) weights for end-to-end inference.",
    )
    parser.add_argument(
        "--vlm_output_dir",
        default="/share/pi05_models/openpi05_base_pytorch",
        help="Output directory for OpenPI PyTorch VLM weights.",
    )
    parser.add_argument(
        "--vlm_precision",
        default="float32",
        choices=["bfloat16", "float32"],
        help="Precision for exported OpenPI PyTorch VLM weights.",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint_dir
    params_path = ckpt if ckpt.endswith("/params") else os.path.join(ckpt, "params")
    params = restore_params(params_path, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params, sep="/")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    state, config, bias_warnings = build_state_dict(flat, args.action_horizon, dtype)

    save_file(state, args.output_path)

    report = {
        "checkpoint_dir": ckpt,
        "output_path": args.output_path,
        "config_inferred": config,
        "action_horizon": args.action_horizon,
        "dtype": args.dtype,
        "bias_warnings": bias_warnings,
    }
    report_path = args.output_path + ".report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"OK: saved {args.output_path}")
    print(f"OK: report {report_path}")

    if args.export_vlm:
        vlm_output_dir = Path(args.vlm_output_dir)
        export_openpi_vlm_weights(params, vlm_output_dir, args.vlm_precision)

        assets_src = Path(ckpt) / "assets"
        if assets_src.exists():
            shutil.copytree(assets_src, vlm_output_dir / "assets", dirs_exist_ok=True)
        tokenizer_src = Path(ckpt) / "paligemma_tokenizer.model"
        if tokenizer_src.exists():
            shutil.copy2(tokenizer_src, vlm_output_dir / "paligemma_tokenizer.model")


if __name__ == "__main__":
    main()
