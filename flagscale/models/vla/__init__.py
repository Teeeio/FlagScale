from .action_models.flow_matching import FlowMatchingHead
from .protocols import ActionModel, VLMBackbone
from .registry import (
    ACTION_MODEL_REGISTRY,
    VLM_REGISTRY,
    build_action_model,
    build_vlm,
    register_action_model,
    register_vlm,
)
from .utils import get_vlm_config

# Explicit registration
from .vlm.qwen_vl import Qwen3VLBackbone, Qwen25VLBackbone

VLM_REGISTRY["qwen2.5-vl"] = Qwen25VLBackbone
VLM_REGISTRY["qwen3-vl"] = Qwen3VLBackbone
ACTION_MODEL_REGISTRY["flow_matching"] = FlowMatchingHead

__all__ = [
    "VLMBackbone",
    "ActionModel",
    "register_vlm",
    "register_action_model",
    "build_vlm",
    "build_action_model",
    "get_vlm_config",
]
