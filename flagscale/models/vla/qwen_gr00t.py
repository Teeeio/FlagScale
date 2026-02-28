# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/framework/QwenGR00T.py
# Below is the original copyright:

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.registry import build_action_model, build_vlm
from flagscale.train.train_config import TrainConfig

# Subdirectory within checkpoint for saved VLM architecture config + processor
VLM_CONFIG_DIR = "vlm_config"


class QwenGr00t(TrainablePolicy):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__()
        self._config = config

        vlm_type = config.model.vlm.get("type", "qwen3-vl")
        self.vlm = build_vlm(vlm_type, config=config)

        action_model_type = config.model.action_model.get("type", "flow_matching")
        self.action_model = build_action_model(
            action_model_type,
            vlm_config=self.vlm.model_config,
            action_config={},
            full_config=config,
        )

        self.future_action_window_size = config.model.action_model.future_action_window_size

        # Deserialize input/output features from config (checkpoint load path).
        # At training time, make_policy sets these on the policy after construction.
        load_pretrained = config.model.qwenvl.get("load_pretrained", True)
        raw_input = config.model.get("input_features", {})
        raw_output = config.model.get("output_features", {})

        if not load_pretrained and (not raw_input or not raw_output):
            raise ValueError(
                "Checkpoint config missing input_features/output_features. "
                "Re-save the checkpoint with the latest training code."
            )

        if raw_input:
            self.input_features = {
                k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in raw_input.items()
            }
        if raw_output:
            self.output_features = {
                k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in raw_output.items()
            }

    def forward(self, examples: dict, **kwargs):
        """ """
        # actions = [example["action"] for example in examples]  # [B, T, action_dim]
        actions = examples[ACTION]
        state = None  # examples[OBS_STATE]

        # Step 1: QWenVL input format
        # NOTE: (yupu) The order of the images differs from starVLA, which is [image, wrist_image]
        qwen_inputs = self.vlm.prepare_input(
            examples, image_feature_keys=list(self.image_features.keys())
        )

        # TODO: (yupu) Hard-coded autocast and dtype, matches starVLA
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # TODO: (yupu) Is this a bug or a feature? The action dtype would stay as bf16 under this autocast.
            actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
            # actions = torch.tensor(
            #     np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            # )  # [B, T_full, action_dim]

            # TODO: does not match RoboBrainX, need to check
            actions_target = actions[
                :, -(self.future_action_window_size + 1) :, :
            ]  # (B, chunk_len, action_dim)

            # TODO: (yupu) I believe there is a bug in starVLA, the
            # `repeated_diffusion_steps` is not properly set in the config.
            repeated_diffusion_steps = self._config.model.action_model.get(
                "repeated_diffusion_steps", 4
            )

            actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # Use action head forward API
            vlm_output_repeated = {"hidden_states": last_hidden_repeated}
            action_input = {"actions": actions_repeated, "state": state_repeated}

            output = self.action_model.forward(vlm_output_repeated, action_input)

        return {"loss": output["loss"]}

    @torch.inference_mode()
    def predict_action(self, examples: list[dict], **kwargs) -> dict:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        # TODO: (yupu) Fix inference input format to use constants (OBS_IMAGE, OBS_LANGUAGE, OBS_STATE)
        # instead of hardcoded keys. The current keys are inconsistent with training batch format.
        # batch_images = [[to_pil_preserve(example["image"])] for example in examples]  # [B, [PLT]]
        # instructions = [example["lang"] for example in examples]  # [B, str]

        # We assume the images are already resized during preprocessing.
        qwen_inputs = self.vlm.prepare_input(
            examples, image_feature_keys=list(self.image_features.keys())
        )
        state = None  # examples[OBS_STATE]

        # state = (
        #     [example["state"] for example in examples] if "state" in examples[0] else None
        # )  # [B, 1, state_dim]

        # train_obs_image_size = getattr(self._config.data.vla_data, "image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # # Step 1: QWenVL input format
        # qwen_inputs = self.vlm.build_qwenvl_inputs(
        #     examples=None, images=batch_images, instructions=instructions
        # )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        if state is not None:
            state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)

        # state_tensor = (
        #     torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
        #     if state is not None
        #     else None
        # )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            vlm_output_for_action = {"hidden_states": last_hidden}
            action_input = {"state": state}
            output = self.action_model.predict_action(vlm_output_for_action, action_input)

        # Assume the output of the action model is dict mapping `ACTION` to the normalized actions
        return output

    def save_pretrained_configs(self, save_dir: Path) -> None:
        vlm_config_dir = save_dir / VLM_CONFIG_DIR
        vlm_config_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.model.config.save_pretrained(vlm_config_dir)
        self.vlm.processor.save_pretrained(vlm_config_dir)

        # Patch saved train config so load_checkpoint skips pretrained VLM download
        config_path = save_dir / "train_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Expected train_config.yaml at {config_path}. "
                "save_pretrained_configs must be called after save_checkpoint."
            )
        saved_config = OmegaConf.load(config_path)
        OmegaConf.update(saved_config, "model.qwenvl.load_pretrained", False)
        OmegaConf.update(saved_config, "model.qwenvl.base_vlm", str(vlm_config_dir))
        OmegaConf.update(
            saved_config,
            "model.input_features",
            {
                k: {"type": ft.type.value, "shape": list(ft.shape)}
                for k, ft in self.input_features.items()
            },
        )
        OmegaConf.update(
            saved_config,
            "model.output_features",
            {
                k: {"type": ft.type.value, "shape": list(ft.shape)}
                for k, ft in self.output_features.items()
            },
        )
        OmegaConf.save(saved_config, config_path)
