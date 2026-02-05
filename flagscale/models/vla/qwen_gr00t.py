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

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from flagscale.models.vla.registry import build_action_model, build_vlm
from flagscale.train.train_config import TrainConfig
from flagscale.train.utils.image_tools import to_pil_preserve
from flagscale.train.utils.trainer_tools import resize_images


class QwenGr00t(PreTrainedModel):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    config_class = PretrainedConfig

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__(PretrainedConfig())
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

        # DEBUG: Print action encoder weights to verify initialization matches starVLA
        if hasattr(self.action_model, "_head") and hasattr(
            self.action_model._head, "action_encoder"
        ):
            ae = self.action_model._head.action_encoder
            print(
                f"[DEBUG INIT] action_encoder.layer1.weight[:3,:5]: {ae.layer1.weight[:3, :5].tolist()}"
            )
            print(
                f"[DEBUG INIT] action_encoder.layer1.weight sum: {ae.layer1.weight.sum().item():.6f}"
            )

    def forward(self, examples: dict, **kwargs):
        """ """
        actions = [example["action"] for example in examples]  # [B, T, action_dim]
        # actions = examples[ACTION]
        state = None  # examples[OBS_STATE]

        # Step 1: QWenVL input format
        # NOTE: (yupu) The order of the images differs from starVLA, which is [image, wrist_image]
        qwen_inputs = self.vlm.prepare_input(examples)

        # DEBUG: Print qwen_inputs stats
        print(f"[DEBUG] qwen_inputs keys: {qwen_inputs.keys()}")
        print(f"[DEBUG] input_ids shape: {qwen_inputs['input_ids'].shape}")
        print(f"[DEBUG] input_ids sum: {qwen_inputs['input_ids'].sum().item()}")

        # qwen_inputs = torch.load("/share/project/fengyupu/github/starVLA/qwen_inputs_debug.pt", weights_only=False)
        # torch.testing.assert_close(qwen_inputs, qwen_inputs_debug)

        # torch.save(qwen_inputs, "qwen_inputs.pt")

        # TODO: (yupu) Hard-coded autocast and dtype, matches starVLA
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]
            print(f"[DEBUG] last_hidden shape: {last_hidden.shape}, dtype: {last_hidden.dtype}")
            print(
                f"[DEBUG] last_hidden norm: {last_hidden.norm().item():.4f}, mean: {last_hidden.mean().item():.6f}, std: {last_hidden.std().item():.6f}"
            )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # TODO: (yupu) Is this a bug or a feature? The action dtype would stay as bf16 under this autocast.
            # actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]

            # TODO: does not match RoboBrainX, need to check
            actions_target = actions[
                :, -(self.future_action_window_size + 1) :, :
            ]  # (B, chunk_len, action_dim)

            # TODO: (yupu) I believe there is a bug in starVLA, the
            # `repeated_diffusion_steps` is not properly set in the config.
            repeated_diffusion_steps = self._config.model.action_model.get(
                "repeated_diffusion_steps", 4
            )

            print(f"[DEBUG] actions_target shape before repeat: {actions_target.shape}")
            print(f"[DEBUG] actions_target sum: {actions_target.sum().item():.4f}")
            print(f"[DEBUG] actions_target[0,0,:5]: {actions_target[0, 0, :5].tolist()}")
            print(f"[DEBUG] repeated_diffusion_steps: {repeated_diffusion_steps}")

            actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            print(f"[DEBUG] actions_repeated shape: {actions_repeated.shape}")
            print(f"[DEBUG] last_hidden_repeated shape: {last_hidden_repeated.shape}")

            state_repeated = None
            if state is not None:
                state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # Use action head forward API
            vlm_output_repeated = {"hidden_states": last_hidden_repeated}
            action_input = {"actions": actions_repeated, "state": state_repeated}

            # torch.save(vlm_output_repeated, "vlm_output_repeated.pt")
            # torch.save(action_input, "action_input.pt")

            output = self.action_model.forward(vlm_output_repeated, action_input)

            # torch.save(output, "output.pt")

        print(f"output: {output}")
        # assert False

        return output["loss"]

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
        if type(examples) is not list:
            examples = [examples]
        batch_images = [[to_pil_preserve(example["image"])] for example in examples]  # [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # [B, 1, state_dim]

        train_obs_image_size = getattr(self._config.data.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.vlm.build_qwenvl_inputs(
            examples=None, images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        state_tensor = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            vlm_output_for_action = {"hidden_states": last_hidden}
            action_input = {"state": state_tensor}
            output = self.action_model.predict(vlm_output_for_action, action_input)

        return {"normalized_actions": output["actions"].detach().cpu().numpy()}
