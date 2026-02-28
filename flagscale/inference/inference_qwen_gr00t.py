import argparse
import importlib

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from flagscale.logger import logger
from flagscale.models.utils.constants import OBS_STATE
from flagscale.train.utils.train_utils import load_checkpoint


def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def load_state_from_file(state_path: str) -> torch.Tensor:
    # (1, state_dim)
    state = torch.load(state_path, map_location="cpu")
    return state


def run_inference(config_path: str):
    logger.info(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    engine_cfg = cfg.engine
    generate_cfg = cfg.generate

    # TODO: (yupu) Use `PreTrainedModel` for save/load
    model_variant = engine_cfg.model_variant
    policy = getattr(importlib.import_module("flagscale.models.vla"), model_variant)
    model, preprocessor, postprocessor = load_checkpoint(
        engine_cfg.model, policy, engine_cfg.device
    )

    # TODO: (yupu): model.to(dtype)?

    # FIXME: images are not resized
    images = generate_cfg.images
    state_path = generate_cfg.get("state_path")
    task_path = generate_cfg.get("task_path")

    image_keys = list(images.keys())
    logger.info(f"Loading {len(image_keys)} images...")
    loaded_images = {}
    for img_key, img_path in images.items():
        img = load_image(img_path)
        loaded_images[img_key] = img
        logger.info(f"Loaded image: {img_key} from {img_path} with shape {img.shape}")

    logger.info(f"Loading state from {state_path}...")
    state = load_state_from_file(state_path)
    logger.info(f"Loaded state with shape: {state.shape}")

    logger.info(f"Loading task from {task_path}...")
    with open(task_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    logger.info(f"Loaded task prompt: '{prompt}'")

    batch = {}
    for img_key, img in loaded_images.items():
        batch[img_key] = img
    batch[OBS_STATE] = state
    batch["task"] = [prompt]

    # batch = {
    #     k: v.to(engine_cfg.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    #     for k, v in batch.items()
    # }

    logger.info("Preprocessing batch...")
    batch = preprocessor(batch)

    logger.info("Running inference...")
    with torch.no_grad():
        action = model.predict_action(batch)
        logger.info(f"action before postprocessor: {action}")

    logger.info("Applying postprocessor...")
    action = postprocessor(action)
    logger.info(f"action after postprocessor: {action}")

    logger.info(f"Final action: {action}")

    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to config YAML file")

    args = parser.parse_args()
    run_inference(config_path=args.config_path)


if __name__ == "__main__":
    main()
