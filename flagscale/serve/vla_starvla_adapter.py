from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION

STARVLA_IMAGE_KEY = "observation/image"
STARVLA_WRIST_IMAGE_KEY = "observation/wrist_image"
STARVLA_RIGHT_IMAGE_KEY = "observation/image_right"
STARVLA_STATE_KEY = "observation/state"
STARVLA_PROMPT_KEY = "prompt"
STARVLA_ACTIONS_KEY = "actions"
INTERNAL_TASK_KEY = "task"

MissingImagePolicy = Literal["error", "black"]
SlotName = Literal["main", "wrist", "right"]

_STARVLA_SLOT_KEYS: dict[SlotName, str] = {
    "main": STARVLA_IMAGE_KEY,
    "wrist": STARVLA_WRIST_IMAGE_KEY,
    "right": STARVLA_RIGHT_IMAGE_KEY,
}


@dataclass(frozen=True)
class PolicyInputLayout:
    visual_slot_map: dict[SlotName, str]
    state_key: str | None
    image_hw: tuple[int, int] | None


def extract_policy_input_features(model: Any, preprocessor: Any | None) -> dict[str, PolicyFeature]:
    input_features = getattr(model, "input_features", None)
    if input_features:
        return dict(input_features)

    model_config = getattr(model, "config", None)
    config_features = getattr(model_config, "input_features", None)
    if config_features:
        return dict(config_features)

    if preprocessor is not None:
        for step in getattr(preprocessor, "steps", []):
            step_features = getattr(step, "features", None)
            if step_features:
                return {
                    key: feature
                    for key, feature in step_features.items()
                    if feature.type != FeatureType.ACTION
                }

    raise ValueError("Unable to resolve input features from model or checkpoint preprocessor.")


def resolve_policy_image_hw(
    train_config: Any | None,
    input_features: Mapping[str, PolicyFeature],
    config_image_hw: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if config_image_hw is not None:
        return config_image_hw

    default_resolution = _resolve_default_image_resolution(train_config)

    if default_resolution and len(default_resolution) >= 3:
        return int(default_resolution[-2]), int(default_resolution[-1])

    for feature in input_features.values():
        if feature.type != FeatureType.VISUAL or len(feature.shape) < 2:
            continue
        shape = tuple(int(dim) for dim in feature.shape)
        if len(shape) >= 3 and shape[0] in (1, 3, 4):
            return shape[-2], shape[-1]
        return shape[0], shape[1]

    return None


def resolve_policy_input_layout(
    input_features: Mapping[str, PolicyFeature],
    *,
    state_key: str | None = None,
    image_hw: tuple[int, int] | None = None,
    explicit_visual_slot_map: Mapping[str, str] | None = None,
) -> PolicyInputLayout:
    visual_keys = [
        key for key, feature in input_features.items() if feature.type == FeatureType.VISUAL
    ]
    if len(visual_keys) > 3:
        raise ValueError(
            f"starVLA simulation protocol supports at most 3 visual inputs, got {len(visual_keys)}: {visual_keys}"
        )

    slot_map = (
        _normalize_explicit_slot_map(explicit_visual_slot_map, visual_keys)
        if explicit_visual_slot_map
        else _infer_visual_slot_map(visual_keys)
    )

    resolved_state_key = state_key
    if resolved_state_key is None:
        for key, feature in input_features.items():
            if feature.type == FeatureType.STATE:
                resolved_state_key = key
                break

    return PolicyInputLayout(
        visual_slot_map=slot_map,
        state_key=resolved_state_key,
        image_hw=image_hw,
    )


class StarVLASimAdapter:
    def __init__(
        self,
        input_layout: PolicyInputLayout,
        *,
        missing_image_policy: MissingImagePolicy = "error",
    ) -> None:
        if missing_image_policy not in {"error", "black"}:
            raise ValueError(
                f"Invalid missing_image_policy={missing_image_policy!r}; expected 'error' or 'black'."
            )
        self.input_layout = input_layout
        self.missing_image_policy = missing_image_policy

    def adapt_request(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        payload = self._unwrap_payload(observation)
        batch: dict[str, Any] = {}

        # Preserve already-canonical internal keys for debugging/manual usage.
        for target_key in self.input_layout.visual_slot_map.values():
            if target_key in payload:
                batch[target_key] = self._prepare_image(payload[target_key])

        if self.input_layout.state_key and self.input_layout.state_key in payload:
            batch[self.input_layout.state_key] = self._prepare_state(
                payload[self.input_layout.state_key]
            )

        if INTERNAL_TASK_KEY in payload:
            batch[INTERNAL_TASK_KEY] = _normalize_prompt(payload[INTERNAL_TASK_KEY])

        # Normalize starVLA protocol keys into the checkpoint's internal feature keys
        # before the serialized preprocessor runs so historical checkpoints stay usable.
        for slot_name, target_key in self.input_layout.visual_slot_map.items():
            if target_key in batch:
                continue
            source_key = _STARVLA_SLOT_KEYS[slot_name]
            if source_key in payload:
                batch[target_key] = self._prepare_image(payload[source_key])
                continue

            if self.missing_image_policy == "black":
                batch[target_key] = self._black_image()
                continue

            raise ValueError(
                f"Missing required starVLA image key '{source_key}' for model feature '{target_key}'."
            )

        if (
            self.input_layout.state_key
            and self.input_layout.state_key not in batch
            and STARVLA_STATE_KEY in payload
        ):
            batch[self.input_layout.state_key] = self._prepare_state(payload[STARVLA_STATE_KEY])

        if INTERNAL_TASK_KEY not in batch:
            if STARVLA_PROMPT_KEY not in payload:
                raise ValueError(
                    f"Missing required prompt key '{STARVLA_PROMPT_KEY}' (or internal '{INTERNAL_TASK_KEY}')."
                )
            batch[INTERNAL_TASK_KEY] = _normalize_prompt(payload[STARVLA_PROMPT_KEY])

        return batch

    def format_response(self, action_output: Any) -> dict[str, list[list[float]]]:
        if isinstance(action_output, dict):
            if ACTION not in action_output:
                raise ValueError(f"Model output dict is missing '{ACTION}'.")
            action_output = action_output[ACTION]

        action_tensor = torch.as_tensor(action_output)
        if action_tensor.ndim == 3:
            if action_tensor.shape[0] != 1:
                raise ValueError(
                    f"Serving only supports batch size 1, got action tensor shape {tuple(action_tensor.shape)}."
                )
            action_tensor = action_tensor[0]
        elif action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim != 2:
            raise ValueError(f"Unexpected action tensor shape {tuple(action_tensor.shape)}.")

        return {STARVLA_ACTIONS_KEY: action_tensor.detach().cpu().to(torch.float32).tolist()}

    def _prepare_state(self, value: Any) -> torch.Tensor:
        state = np.asarray(value)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1:
            raise ValueError(f"Expected 1D state vector, got shape {state.shape}.")
        return torch.from_numpy(np.array(state, copy=True, dtype=np.float32))

    def _prepare_image(self, value: Any) -> torch.Tensor:
        image = _to_numpy_hwc_uint8(value)
        if self.input_layout.image_hw is None:
            return _image_to_tensor(image)
        target_h, target_w = self.input_layout.image_hw
        if image.shape[:2] == (target_h, target_w):
            return _image_to_tensor(image)
        resized = np.asarray(
            Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR)
        )
        return _image_to_tensor(resized)

    def _black_image(self) -> torch.Tensor:
        if self.input_layout.image_hw is None:
            raise ValueError(
                "missing_image_policy='black' requires a resolvable target image size, but no image size was found."
            )
        target_h, target_w = self.input_layout.image_hw
        return torch.zeros((target_h, target_w, 3), dtype=torch.uint8)

    @staticmethod
    def _unwrap_payload(observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if "examples" in observation:
            examples = observation["examples"]
            if not isinstance(examples, (list, tuple)) or len(examples) != 1:
                raise ValueError("Expected 'examples' to contain exactly one sample for serving.")
            sample = examples[0]
            if not isinstance(sample, Mapping):
                raise ValueError("Expected the single 'examples' element to be a mapping.")
            return sample
        return observation


def _normalize_explicit_slot_map(
    explicit_visual_slot_map: Mapping[str, str],
    visual_keys: list[str],
) -> dict[SlotName, str]:
    normalized: dict[SlotName, str] = {}
    unknown = set(explicit_visual_slot_map.values()) - set(visual_keys)
    if unknown:
        raise ValueError(
            f"Explicit starVLA slot map references unknown visual feature keys: {sorted(unknown)}."
        )
    for raw_slot, feature_key in explicit_visual_slot_map.items():
        slot = raw_slot.lower()
        if slot not in _STARVLA_SLOT_KEYS:
            raise ValueError(
                f"Unsupported starVLA slot name {raw_slot!r}; expected one of {sorted(_STARVLA_SLOT_KEYS)}."
            )
        normalized[slot] = feature_key
    return normalized


def _infer_visual_slot_map(visual_keys: list[str]) -> dict[SlotName, str]:
    remaining = list(visual_keys)
    resolved: dict[SlotName, str] = {}

    for slot_name, predicate in (
        ("right", _looks_like_right_image),
        ("wrist", _looks_like_wrist_image),
        ("main", _looks_like_main_image),
    ):
        for key in remaining:
            if predicate(key):
                resolved[slot_name] = key
                remaining.remove(key)
                break

    for slot_name in ("main", "wrist", "right"):
        if slot_name not in resolved and remaining:
            resolved[slot_name] = remaining.pop(0)

    return resolved


def _resolve_default_image_resolution(train_config: Any | None) -> Any | None:
    if train_config is None:
        return None
    if OmegaConf.is_config(train_config):
        return OmegaConf.select(train_config, "data.vla_data.default_image_resolution")
    if isinstance(train_config, Mapping):
        data = train_config.get("data")
        if isinstance(data, Mapping):
            vla_data = data.get("vla_data")
            if isinstance(vla_data, Mapping):
                return vla_data.get("default_image_resolution")
        return None
    data = getattr(train_config, "data", None)
    vla_data = getattr(data, "vla_data", None)
    return getattr(vla_data, "default_image_resolution", None)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(image, copy=True, dtype=np.uint8))


def _normalize_prompt(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected a single prompt string, got {len(value)} values.")
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"Expected prompt/task to be a string, got {type(value).__name__}.")
    return value


def _looks_like_main_image(key: str) -> bool:
    key = key.lower()
    return (
        "wrist" not in key
        and "right" not in key
        and any(
            token in key for token in ("base", "front", "high", ".image", "_image", "agentview")
        )
    )


def _looks_like_wrist_image(key: str) -> bool:
    key = key.lower()
    return any(token in key for token in ("wrist", "hand", "eye_in_hand"))


def _looks_like_right_image(key: str) -> bool:
    return "right" in key.lower()


def _to_numpy_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        image = np.asarray(value.convert("RGB"))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected single-image tensor, got shape {tuple(tensor.shape)}.")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D image tensor, got shape {tuple(tensor.shape)}.")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
    else:
        image = np.asarray(value)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {image.shape}.")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = (
            np.clip(image, 0.0, 1.0) * 255.0 if np.issubdtype(image.dtype, np.floating) else image
        )
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image
