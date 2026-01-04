"""JAX-aligned SigLIP vision wrapper for Pi0.5.

This module avoids OpenPI runtime dependencies by using a vendored SigLIP
implementation and loading weights from converted OpenPI checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from safetensors.torch import load_file
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from flagscale.models.pi05.siglip_jax import SiglipVisionModel


@dataclass
class SiglipJaxVisionConfig:
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    image_size: int = 224
    patch_size: int = 14
    projection_dim: int = 2048
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0


class SiglipJaxVisionWithProjector(nn.Module):
    def __init__(self, config: SiglipJaxVisionConfig | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        cfg = config or SiglipJaxVisionConfig()
        vision_cfg = SiglipVisionConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            layer_norm_eps=cfg.layer_norm_eps,
            hidden_act=cfg.hidden_act,
            attention_dropout=cfg.attention_dropout,
        )
        self.vision_model = SiglipVisionModel(vision_cfg)
        self.projector = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=True)
        self.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        tokens = outputs.last_hidden_state
        return self.projector(tokens)

    def load_openpi_pytorch_weights(self, weights_path: str) -> None:
        state = load_file(weights_path)
        vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower."
        projector_weight = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
        projector_bias = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"

        vision_state = {}
        for key, value in state.items():
            if key.startswith(vision_prefix):
                new_key = key[len(vision_prefix):]
                vision_state[new_key] = value

        missing, unexpected = self.vision_model.load_state_dict(vision_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected vision keys: {unexpected}")
        if missing:
            missing_allowed = [k for k in missing if k.startswith("vision_model.head.")]
            if len(missing_allowed) != len(missing):
                raise ValueError(f"Missing vision keys: {missing}")

        if projector_weight not in state or projector_bias not in state:
            raise ValueError("Missing multi_modal_projector weights in checkpoint")
        self.projector.weight.data.copy_(state[projector_weight])
        self.projector.bias.data.copy_(state[projector_bias])
