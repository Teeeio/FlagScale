import argparse
from typing import Union

import torch

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.inference.utils import parse_torch_dtype
from flagscale.models.pi0.modeling_pi0 import PI0Policy, PI0PolicyConfig
from flagscale.models.pi05 import Pi05Model
from flagscale.models.pi05.mask_utils import expand_image_masks_from_tokens
from flagscale.models.pi05.siglip_jax_vision import SiglipJaxVisionWithProjector
from flagscale.runner.utils import logger


def parse_config() -> Union[DictConfig, ListConfig]:
    """Parse the configuration file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    return OmegaConf.load(args.config_path)


def build_input(generate_cfg, dtype, device):
    batch = {}
    batch_size = generate_cfg.batch_size
    for k in generate_cfg.images_keys:
        batch[k] = torch.randn(
            batch_size, *generate_cfg.images_shape, dtype=dtype, device=device
        )
    state_dim = int(getattr(generate_cfg, "state_dim", generate_cfg.action_dim))
    batch[generate_cfg.state_key] = torch.randn(batch_size, state_dim, dtype=dtype, device=device)
    batch.update(generate_cfg.instruction)
    return batch


def resize_with_pad_torch(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize with padding to keep aspect ratio, OpenPI-compatible."""
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = torch.nn.functional.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        constant_value = 0
    else:
        resized = resized.clamp(-1.0, 1.0)
        constant_value = -1.0

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded = torch.nn.functional.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value
    )

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
        if batch_size == 1 and images.shape[0] == 1:
            padded = padded.squeeze(0)
    return padded


def inference(cfg: DictConfig) -> None:
    """Pi0.5 inference entrypoint aligned to Pi0 preprocessing."""
    generate_cfg = cfg.get("generate", {})
    if isinstance(generate_cfg, dict):
        generate_cfg = OmegaConf.create(generate_cfg)
    engine_cfg = cfg.get("engine", {})
    if isinstance(engine_cfg, dict):
        engine_cfg = OmegaConf.create(engine_cfg)
    dtype_config = engine_cfg.get("torch_dtype")
    dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32

    use_jax_siglip = bool(engine_cfg.get("pi05_siglip_weights"))
    pi0_policy = None
    if engine_cfg.get("pi0_model") and engine_cfg.get("pi0_tokenizer") and engine_cfg.get("pi0_stat"):
        pi0_cfg = PI0PolicyConfig.from_pretrained(engine_cfg.pi0_model)
        pi0_policy = PI0Policy.from_pretrained(
            model_path=engine_cfg.pi0_model,
            tokenizer_path=engine_cfg.pi0_tokenizer,
            stat_path=engine_cfg.pi0_stat,
            config=pi0_cfg,
        ).to(device=engine_cfg.device)
        pi0_policy.eval()

    # Pi0.5 model
    model_cfg = dict(engine_cfg.get("pi05_config", {}))
    pi05_model = Pi05Model(**model_cfg).to(device=engine_cfg.device)
    weights_path = engine_cfg.get("pi05_weights") or engine_cfg.get("model")
    if weights_path:
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        pi05_model.load_state_dict(state_dict, strict=False)
    pi05_model.eval()

    batch = build_input(generate_cfg, dtype, engine_cfg.device)

    lang_tokens = None
    lang_masks = None
    if pi0_policy is not None:
        lang_tokens, lang_masks = pi0_policy.prepare_language(batch)

    if use_jax_siglip:
        siglip = SiglipJaxVisionWithProjector(dtype=torch.float32).to(device=engine_cfg.device)
        siglip.load_openpi_pytorch_weights(engine_cfg.pi05_siglip_weights)
        siglip.eval()

        images = []
        img_masks = []
        for key in generate_cfg.images_keys:
            img = batch[key]
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0 * 2.0 - 1.0
            else:
                img = img.to(torch.float32) * 2.0 - 1.0
            img = resize_with_pad_torch(img, 224, 224)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.shape[1] != 3:
                img = img.permute(0, 3, 1, 2)
            images.append(img.to(device=engine_cfg.device))
            img_masks.append(torch.ones((img.shape[0],), dtype=torch.bool, device=engine_cfg.device))

        image_tokens = [siglip(img) for img in images]
        image_mask = expand_image_masks_from_tokens(img_masks, image_tokens)
        image_tokens = torch.cat(image_tokens, dim=1).to(dtype=pi05_model.dtype)
        logger.info(f"image_tokens: {tuple(image_tokens.shape)} image_mask: {tuple(image_mask.shape)}")
    else:
        if pi0_policy is None:
            raise ValueError("pi0_model/pi0_tokenizer/pi0_stat are required unless pi05_siglip_weights is provided.")
        images, img_masks = pi0_policy.prepare_images(batch)
        images = [i.to(dtype=dtype) for i in images]

        # Encode images with Pi0 VLM, then build image_mask aligned to tokens
        image_tokens = []
        for img in images:
            img_emb = pi0_policy.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            img_emb = img_emb * torch.tensor(
                img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            image_tokens.append(img_emb)

        image_mask = expand_image_masks_from_tokens(img_masks, image_tokens)
        image_tokens = torch.cat(image_tokens, dim=1)

    with torch.no_grad():
        actions = pi05_model.sample_actions(
            image_tokens=image_tokens,
            language_tokens=lang_tokens,
            image_mask=image_mask,
            language_mask=lang_masks,
            num_steps=generate_cfg.num_steps,
        )
        logger.info(f"actions: {actions.shape}")

    actions_trunked = actions[:, : generate_cfg.action_horizon, : generate_cfg.action_dim]
    logger.info(f"actions_trunked: {actions_trunked}")


if __name__ == "__main__":
    parsed_cfg = parse_config()
    assert isinstance(parsed_cfg, DictConfig)
    inference(parsed_cfg)
    logger.info("Pi0.5 inference entrypoint completed")
