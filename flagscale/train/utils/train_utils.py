# Copied from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/utils/train_utils.py

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_model, save_model

from flagscale.models.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STEP,
)
from flagscale.train.datasets.utils import load_json, write_json

# from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    policy,
    config,
    preprocessor=None,
) -> None:
    """Save model weights, config, and preprocessor state.

    Creates the following directory structure:
        005000/
        └── pretrained_model/
            ├── train_config.yaml              # train config (OmegaConf)
            ├── model.safetensors              # All weights (VLM + action head)
            ├── policy_preprocessor.json       # Preprocessor pipeline config
            └── policy_preprocessor_step_*.safetensors  # Norm stats

    Args:
        checkpoint_dir: Directory to save checkpoint (e.g., checkpoints/005000)
        policy: The model
        config: Training config (OmegaConf, Pydantic, or dict)
        preprocessor: Optional PolicyProcessorPipeline
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Save train config as YAML
    # Handle OmegaConf, Pydantic, and dict configs
    if hasattr(config, "model_dump"):
        config = OmegaConf.create(config.model_dump())
    elif not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    OmegaConf.save(config, pretrained_dir / "train_config.yaml")

    # Save model weights (save_model handles shared tensors like tied embeddings)
    save_model(policy, pretrained_dir / "model.safetensors")

    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)


def load_checkpoint(
    checkpoint_dir: str | Path,
    model_cls,
    device: str = "cpu",
):
    """Load config, model weights, and preprocessor from checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory (e.g., checkpoints/005000)
        model_cls: Model class.
        device: Device to load weights to

    Returns:
        If model_cls provided: tuple of (model, preprocessor)
        If model_cls is None: tuple of (config, state_dict, preprocessor)

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
    """
    from flagscale.train.processor import PolicyProcessorPipeline

    print(f"Loading checkpoint from {checkpoint_dir}")

    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {pretrained_dir}")

    config_path = pretrained_dir / "train_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)

    model = model_cls(config)

    weights_path = pretrained_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    # TODO: (yupu) Some modules could be loaded twice
    missing_keys, unexpected_keys = load_model(model, weights_path, device=device)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            for key in missing_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(missing_keys) - 10} more")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(unexpected_keys) - 10} more")

    model.to(device)

    preprocessor = None
    preprocessor_config_path = pretrained_dir / "policy_preprocessor.json"
    if preprocessor_config_path.exists():
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_preprocessor.json",
        )

    return model, preprocessor


# def save_training_state(
#     checkpoint_dir: Path,
#     train_step: int,
#     optimizer: Optimizer | None = None,
#     scheduler: LRScheduler | None = None,
# ) -> None:
#     """
#     Saves the training step, optimizer state, scheduler state, and rng state.

#     Args:
#         save_dir (Path): The directory to save artifacts to.
#         train_step (int): Current training step.
#         optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
#             Defaults to None.
#         scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
#             Defaults to None.
#     """
#     save_dir = checkpoint_dir / TRAINING_STATE_DIR
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_training_step(train_step, save_dir)
#     save_rng_state(save_dir)
#     if optimizer is not None:
#         save_optimizer_state(optimizer, save_dir)
#     if scheduler is not None:
#         save_scheduler_state(scheduler, save_dir)


# def load_training_state(
#     checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
# ) -> tuple[int, Optimizer, LRScheduler | None]:
#     """
#     Loads the training step, optimizer state, scheduler state, and rng state.
#     This is used to resume a training run.

#     Args:
#         checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
#         optimizer (Optimizer): The optimizer to load the state_dict to.
#         scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

#     Raises:
#         NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

#     Returns:
#         tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
#             state_dict loaded.
#     """
#     training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
#     if not training_state_dir.is_dir():
#         raise NotADirectoryError(training_state_dir)

#     load_rng_state(training_state_dir)
#     step = load_training_step(training_state_dir)
#     optimizer = load_optimizer_state(optimizer, training_state_dir)
#     if scheduler is not None:
#         scheduler = load_scheduler_state(scheduler, training_state_dir)

#     return step, optimizer, scheduler
