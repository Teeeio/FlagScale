from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from flagscale.models.utils.constants import ACTION

CANONICAL_IMAGE_KEY = "observation.image"
CANONICAL_WRIST_IMAGE_KEY = "observation.wrist_image"
CANONICAL_RIGHT_IMAGE_KEY = "observation.image_right"
CANONICAL_STATE_KEY = "observation.state"


@dataclass(frozen=True)
class VLARequestProtocol:
    env_name: str
    image_key: str
    wrist_image_key: str
    right_image_key: str | None
    state_key: str
    prompt_key: str
    actions_key: str


@dataclass(frozen=True)
class QwenGr00tServeContract:
    task_key: str = "task"
    require_right_image: bool = False
    image_hw: tuple[int, int] | None = None


def build_request_protocol(protocol_cfg: Mapping[str, Any] | None) -> VLARequestProtocol:
    if not protocol_cfg:
        raise ValueError(
            "Serving expects engine_args.protocol to define request and response keys."
        )

    return VLARequestProtocol(
        env_name=_normalize_required_key(protocol_cfg.get("env_name"), "protocol.env_name"),
        image_key=_normalize_required_key(protocol_cfg.get("image_key"), "protocol.image_key"),
        wrist_image_key=_normalize_required_key(
            protocol_cfg.get("wrist_image_key"), "protocol.wrist_image_key"
        ),
        right_image_key=_normalize_optional_key(
            protocol_cfg.get("right_image_key"), "protocol.right_image_key"
        ),
        state_key=_normalize_required_key(protocol_cfg.get("state_key"), "protocol.state_key"),
        prompt_key=_normalize_required_key(protocol_cfg.get("prompt_key"), "protocol.prompt_key"),
        actions_key=_normalize_required_key(
            protocol_cfg.get("actions_key"), "protocol.actions_key"
        ),
    )


def build_qwen_gr00t_serve_contract(
    *,
    task_key: str | None = None,
    require_right_image: bool = False,
    image_hw: tuple[int, int] | None = None,
) -> QwenGr00tServeContract:
    return QwenGr00tServeContract(
        task_key=_normalize_key(task_key, "task_key", "task"),
        require_right_image=bool(require_right_image),
        image_hw=image_hw,
    )


class VLAProtocolAdapter:
    def __init__(
        self, protocol: VLARequestProtocol, serve_contract: QwenGr00tServeContract
    ) -> None:
        self.protocol = protocol
        self.serve_contract = serve_contract

    def adapt_request(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise ValueError("Serving payload must be a mapping.")

        batch = {
            CANONICAL_IMAGE_KEY: self._prepare_image(
                _require_value(observation, self.protocol.image_key)
            ),
            CANONICAL_WRIST_IMAGE_KEY: self._prepare_image(
                _require_value(observation, self.protocol.wrist_image_key)
            ),
            CANONICAL_STATE_KEY: self._prepare_state(
                _require_value(observation, self.protocol.state_key)
            ),
            self.serve_contract.task_key: _normalize_prompt(
                _require_value(observation, self.protocol.prompt_key)
            ),
        }

        if self.serve_contract.require_right_image:
            if self.protocol.right_image_key is None:
                raise ValueError(
                    "Serving expects engine_args.protocol.right_image_key when the serving "
                    "config requires a right image."
                )
            batch[CANONICAL_RIGHT_IMAGE_KEY] = self._prepare_image(
                _require_value(observation, self.protocol.right_image_key)
            )

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
                    "Serving only supports batch size 1, "
                    f"got action tensor shape {tuple(action_tensor.shape)}."
                )
            action_tensor = action_tensor[0]
        elif action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim != 2:
            raise ValueError(f"Unexpected action tensor shape {tuple(action_tensor.shape)}.")

        return {self.protocol.actions_key: action_tensor.detach().cpu().to(torch.float32).tolist()}

    def _prepare_state(self, value: Any) -> torch.Tensor:
        state = np.asarray(value)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1:
            raise ValueError(f"Expected 1D state vector, got shape {state.shape}.")
        return torch.from_numpy(np.array(state, copy=True, dtype=np.float32))

    def _prepare_image(self, value: Any) -> torch.Tensor:
        image = _to_numpy_hwc_uint8(value)
        if self.serve_contract.image_hw is None:
            return _image_to_tensor(image)

        target_h, target_w = self.serve_contract.image_hw
        if image.shape[:2] != (target_h, target_w):
            image = np.asarray(
                Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR)
            )
        return _image_to_tensor(image)


def _require_value(observation: Mapping[str, Any], source_key: str) -> Any:
    if source_key not in observation:
        raise ValueError(f"Missing required serving key '{source_key}'.")
    return observation[source_key]


def _normalize_required_key(value: str | None, key_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Serving expects engine_args.{key_name} to be a non-empty string.")
    return value


def _normalize_key(value: str | None, key_name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"Serving expects engine_args.{key_name} to be a non-empty string.")
    return value


def _normalize_optional_key(value: str | None, key_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Serving expects engine_args.{key_name} to be a non-empty string.")
    return value


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
