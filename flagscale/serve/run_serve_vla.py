import argparse
import importlib
import time

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.logger import logger
from flagscale.models.utils.constants import ACTION
from flagscale.serve.vla_starvla_adapter import (
    StarVLASimAdapter,
    extract_policy_input_features,
    resolve_policy_image_hw,
    resolve_policy_input_layout,
)
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.utils.train_utils import load_checkpoint


class Policy:
    def __init__(self, config: DictConfig | ListConfig):
        self.config_engine = config["engine_args"]

        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.adapter = None

        self.load_model()

    def load_model(self):
        t_s = time.perf_counter()
        model_variant = self.config_engine.model_variant
        policy_cls = getattr(importlib.import_module("flagscale.models.vla"), model_variant)
        self.model, self.preprocessor, self.postprocessor = load_checkpoint(
            self.config_engine.model, policy_cls, self.config_engine.device
        )

        input_features = extract_policy_input_features(self.model, self.preprocessor)
        # Older checkpoints may only persist feature metadata inside the serialized
        # preprocessor. Recover it here so predict_action can still resolve image keys.
        if hasattr(self.model, "input_features") and not getattr(
            self.model, "input_features", None
        ):
            self.model.input_features = input_features
        image_hw = resolve_policy_image_hw(
            getattr(self.model, "_config", None),
            input_features,
            config_image_hw=_resolve_config_image_hw(self.config_engine),
        )
        input_layout = resolve_policy_input_layout(
            input_features,
            state_key=self.config_engine.get("state_key"),
            image_hw=image_hw,
            explicit_visual_slot_map=self.config_engine.get("starvla_slot_map"),
        )
        self.adapter = StarVLASimAdapter(
            input_layout=input_layout,
            missing_image_policy=self.config_engine.get("missing_image_policy", "error"),
        )

        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")
        logger.info(f"Resolved visual slot map: {self.adapter.input_layout.visual_slot_map}")
        logger.info(f"Resolved state key: {self.adapter.input_layout.state_key}")
        logger.info(f"Resolved image size: {self.adapter.input_layout.image_hw}")

    def inference(self, observation):
        if self.adapter is None:
            raise RuntimeError("Serving adapter is not initialized.")

        logger.info("Start VLA inference")
        batch = self.adapter.adapt_request(observation)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        with torch.no_grad():
            prediction = self.model.predict_action(batch)
            logger.info(f"action before postprocessor: {prediction}")

        response_payload = prediction if isinstance(prediction, dict) else {ACTION: prediction}

        if self.postprocessor is not None:
            logger.info("Applying postprocessor...")
            response_payload = self.postprocessor(response_payload)

        return self.adapter.format_response(response_payload)

    def infer(self, observation):
        return self.inference(observation)


def _resolve_config_image_hw(engine_cfg) -> tuple[int, int] | None:
    image_hw = engine_cfg.get("image_hw")
    if image_hw and len(image_hw) == 2:
        return int(image_hw[0]), int(image_hw[1])

    images_shape = engine_cfg.get("images_shape")
    if images_shape and len(images_shape) >= 3:
        return int(images_shape[-2]), int(images_shape[-1])

    return None


def parse_config() -> DictConfig | ListConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log")
    args = parser.parse_args()
    return OmegaConf.load(args.config_path)


def main(config):
    policy = Policy(config)
    logger.info("Done")
    server = WebsocketPolicyServer(
        policy=policy,
        host=policy.host,
        port=policy.port,
        metadata={"env": "starvla_sim"},
    )
    logger.info("Server running ...")
    server.serve_forever()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"][0])
