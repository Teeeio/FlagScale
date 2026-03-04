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

from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.models.vla.action_model.gr00t_action_header import FlowmatchingActionHead

# from flagscale.models.vlm.qwen2_5_vl import _QWen_VL_Interface
from flagscale.models.vlm.qwen3_vl import _QWen3_VL_Interface
from flagscale.train.utils.image_tools import to_pil_preserve
from flagscale.train.utils.trainer_tools import resize_images


class QwenGR00T(PreTrainedModel):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    config_class = PretrainedConfig

    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__(PretrainedConfig())
        self.config = config
        # self.qwen_vl_interface = _QWen_VL_Interface(config=self.config)
        self.qwen_vl_interface = _QWen3_VL_Interface(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.model.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = FlowmatchingActionHead(
            full_config=self.config
        )  # 修复后续引用

        self.future_action_window_size = config.model.action_model.future_action_window_size
        self.past_action_window_size = config.model.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: list[dict] | None = None,
        **kwargs,
    ) -> tuple:
        """ """
        # FIXME: state is None
        # from torchvision import transforms
        # image_transform = transforms.ToPILImage()

        # batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        # instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]

        actions = examples[ACTION]
        state = examples[OBS_STATE]

        # state = (
        #     [example["state"] for example in examples] if "state" in examples[0] else None
        # )  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            examples=examples,
            image_keys=self.config.data.vla_data.image_features,
            # images=batch_images, instructions=instructions
        )

        # print(f"qwen_inputs: {qwen_inputs}")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # TODO: (yupu) Is this a bug or a feature? The action dtype would stay as bf16 under this autocast.
            if isinstance(actions, torch.Tensor):
                actions = actions.to(device=last_hidden.device, dtype=last_hidden.dtype)
            else:
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )
            # TODO: does not match RoboBrainX, need to check
            # actions = torch.tensor(
            #     np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            # )  # [B, T_full, action_dim]
            actions_target = actions[
                :, -(self.future_action_window_size + 1) :, :
            ]  # (B, chunk_len, action_dim)

            # TODO: (yupu) I believe there is a bug in starVLA, the
            # `repeated_diffusion_steps` is not properly set in the config.
            repeated_diffusion_steps = self.config.model.action_model.get(
                "repeated_diffusion_steps", 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return action_loss

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.data.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
