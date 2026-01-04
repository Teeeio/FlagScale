import argparse
import base64
import io
import time

from pathlib import Path
from typing import Union

import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from flagscale.inference.utils import parse_torch_dtype
from flagscale.models.pi05 import Pi05Model
from flagscale.models.pi05.mask_utils import expand_image_masks_from_tokens
from flagscale.models.pi05.siglip_jax_vision import SiglipJaxVisionWithProjector
from flagscale.models.pi05.tokenizer import PaligemmaTokenizer
from flagscale.runner.utils import logger

app = Flask(__name__)
CORS(app)


class PI05Server:
    def __init__(self, config):
        self.config_generate = OmegaConf.create(config["generate"])
        self.config_engine = OmegaConf.create(config["engine"])

        dtype_config = self.config_engine.get("torch_dtype", "torch.float32")
        self.dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32
        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)
        self.device = self.config_engine.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()
        self.warmup()

    def warmup(self):
        """Warmup the model with dummy input"""
        logger.info("Warming up Pi0.5 model...")
        dummy_batch = self.build_dummy_input()
        _ = self.infer(dummy_batch)
        logger.info("Warmup completed.")

    def load_model(self):
        """Load Pi0.5 model with real weights"""
        t_s = time.time()

        model_cfg = dict(self.config_engine.get("pi05_config", {}))
        if not model_cfg:
            model_cfg = {
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
                "num_layers": 18,
                "vocab_size": 257152,
                "paligemma_width": 2048,
                "action_expert_width": 1024,
                "ffn_dim": 16384,
                "action_expert_ffn_dim": 4096,
                "action_dim": 32,
                "action_horizon": 10,
                "max_seq_len": 8192,
                "dropout": 0.0,
            }

        # Initialize model
        self.model = Pi05Model(**model_cfg)

        # Load pretrained weights if available
        model_path = self.config_engine.get("model")
        if model_path and torch.cuda.is_available():
            try:
                logger.info(f"Loading Pi0.5 weights from: {model_path}")
                if str(model_path).endswith(".safetensors"):
                    from safetensors.torch import load_file

                    state_dict = load_file(model_path)
                else:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    state_dict = checkpoint.get("model", checkpoint)

                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Successfully loaded Pi0.5 weights")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}")
                logger.info("Proceeding with randomly initialized model")

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        self.siglip = None
        siglip_weights = self.config_engine.get("pi05_siglip_weights")
        if siglip_weights:
            self.siglip = SiglipJaxVisionWithProjector(dtype=torch.float32).to(device=self.device)
            self.siglip.load_openpi_pytorch_weights(str(siglip_weights))
            self.siglip.eval()

        self.tokenizer = None
        tokenizer_path = self.config_engine.get("tokenizer")
        if tokenizer_path:
            self.tokenizer = PaligemmaTokenizer(
                max_len=int(self.config_engine.get("max_token_len", 200)),
                tokenizer_path=Path(tokenizer_path),
            )

        logger.info(f"Pi0.5 loaded latency: {time.time() - t_s:.2f}s")

    def build_dummy_input(self):
        """Build dummy input for warmup"""
        batch_size = 1

        batch = {}
        for key in self.config_generate.images_keys:
            batch[key] = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=self.device)
        batch["instruction"] = self.config_generate.instruction["task"][0]
        batch["state"] = torch.zeros(batch_size, self.config_generate.state_dim, device=self.device)
        return batch

    def _prepare_language(self, batch):
        instruction = batch.get("instruction")
        state = batch.get("state")
        if instruction and self.tokenizer is not None:
            state_np = None
            if state is not None:
                state_np = state.detach().cpu().numpy()[0]
            tokens, mask = self.tokenizer.tokenize(instruction, state_np)
            tokens_t = torch.from_numpy(tokens).unsqueeze(0).to(device=self.device, dtype=torch.int64)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device=self.device, dtype=torch.bool)
            return tokens_t, mask_t
        return None, None

    def _prepare_image_tokens(self, batch):
        if self.siglip is None:
            raise RuntimeError("pi05_siglip_weights is required for Pi0.5 serve inference.")
        images = []
        img_masks = []
        for key in self.config_generate.images_keys:
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
            images.append(img.to(device=self.device))
            img_masks.append(torch.ones((img.shape[0],), dtype=torch.bool, device=self.device))

        image_tokens = [self.siglip(img) for img in images]
        image_mask = expand_image_masks_from_tokens(img_masks, image_tokens)
        image_tokens = torch.cat(image_tokens, dim=1).to(dtype=self.model.dtype)
        return image_tokens, image_mask

    def infer(self, batch):
        """Run inference with Pi0.5 model"""
        t_s = time.time()

        lang_tokens, lang_masks = self._prepare_language(batch)
        image_tokens, image_mask = self._prepare_image_tokens(batch)
        num_steps = int(self.config_generate.get("num_steps", 8))

        # Run model forward pass
        with torch.no_grad():
            predicted_actions = self.model.sample_actions(
                image_tokens=image_tokens,
                language_tokens=lang_tokens,
                image_mask=image_mask,
                language_mask=lang_masks,
                num_steps=num_steps,
            )

        # Truncate to desired horizon
        actions_trunked = predicted_actions[
            :, : self.config_generate["action_horizon"], : self.config_generate["action_dim"]
        ]

        logger.info(f"Pi0.5 infer latency: {time.time() - t_s:.2f}s")
        logger.info(f"actions_trunked: {actions_trunked}")

        return actions_trunked

    def serve(self):
        """Start the Flask server"""
        logger.info(f"Pi0.5 Serve URL: http://{self.host}:{self.port}")
        logger.info(f"Available API endpoints:")
        logger.info(f"  - GET  /health  - health check")
        logger.info(f"  - POST /infer   - inference api")
        app.run(host=self.host, port=self.port, debug=False, threaded=True)


PI05_SERVER: PI05Server = None


def decode_image_base64(image_base64):
    """Decode base64 image to tensor"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Resize to expected size (224x224 for Pi0.5)
        image = image.resize((224, 224))

        # Convert to tensor and normalize to [0,1]
        image_array = np.array(image).astype(np.float32) / 255.0

        # Convert to CHW format
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise ValueError(f"Image decode error: {e}")


def process_images(images_json):
    """Process JSON images to tensors"""
    processed = []
    for i, sample in enumerate(images_json):
        try:
            sample_dict = {}
            for k, v in sample.items():
                sample_dict[k] = decode_image_base64(v)
            processed.append(sample_dict)
        except Exception as e:
            logger.error(f"Image[{i}] decode error: {e}")
            raise ValueError(f"Image[{i}] decode error: {e}")
    return processed


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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global PI05_SERVER
    if PI05_SERVER is None:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "error": "Pi0.5 server not initialized",
                }
            ),
            503,
        )

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
        }

    return jsonify(
        {"status": "healthy", "model_loaded": True, "model_type": "Pi0.5", "gpu_info": gpu_info}
    )


@app.route('/infer', methods=['POST'])
def infer_api():
    """Inference API endpoint"""
    global PI05_SERVER
    if PI05_SERVER is None:
        return jsonify({"success": False, "error": "Pi0.5 model not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request format error"}), 400

    # Validate required fields
    required_fields = ['states', 'actions', 'instruction']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return (
            jsonify({"success": False, "error": f"Request requires: {', '.join(missing_fields)}"}),
            400,
        )

    try:
        # Parse inputs
        states = torch.tensor(data['states'], dtype=torch.float32)
        _ = torch.tensor(data['actions'], dtype=torch.float32)
        instruction = data.get('instruction', '')
        images = data.get('images')

        # Move to device
        device = PI05_SERVER.device
        states = states.to(device)

    except Exception as e:
        return jsonify({"success": False, "error": f"Input parameters processing error: {e}"}), 400

    # Process images if provided
    image_batches = {}
    if images is not None:
        try:
            images_tensor = process_images(images)
            if images_tensor and len(images_tensor) > 0:
                first_sample = images_tensor[0]
                fallback_image = None
                if 'cam_high' in first_sample:
                    fallback_image = first_sample['cam_high']
                else:
                    for key in first_sample:
                        if 'image' in key or 'camera' in key:
                            fallback_image = first_sample[key]
                            break
                if fallback_image is not None:
                    for key in PI05_SERVER.config_generate.images_keys:
                        image_batches[key] = fallback_image.unsqueeze(0).to(device)
        except Exception as e:
            return jsonify({"success": False, "error": f"Image processing failed: {e}"}), 400

    # Create dummy image if none provided
    if not image_batches:
        batch_size = states.shape[0]
        for key in PI05_SERVER.config_generate.images_keys:
            image_batches[key] = torch.randn(batch_size, 3, 224, 224, device=device)

    # Run inference
    with torch.no_grad():
        batch = {
            **image_batches,
            "instruction": instruction,
            "state": states,
        }
        predicted_actions = PI05_SERVER.infer(batch)

    return jsonify(
        {
            "success": True,
            "predicted_actions": predicted_actions.tolist(),
            "instruction": instruction,
            "input_shapes": {
                "states": list(states.shape),
                "images_keys": list(PI05_SERVER.config_generate.images_keys),
            },
        }
    )


def parse_config() -> Union[DictConfig, ListConfig]:
    """Parse the configuration file"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log directory")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    return config


def main(config):
    """Main function to start Pi0.5 server"""
    global PI05_SERVER
    PI05_SERVER = PI05Server(config)
    PI05_SERVER.serve()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"])
