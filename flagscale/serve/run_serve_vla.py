import argparse
import importlib
import time
from collections.abc import Mapping

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.logger import logger
from flagscale.models.configs.types import FeatureType
from flagscale.models.utils.constants import ACTION, PRETRAINED_MODEL_DIR
from flagscale.serve.vla_protocol_adapter import (
    CANONICAL_IMAGE_KEY,
    CANONICAL_RIGHT_IMAGE_KEY,
    CANONICAL_STATE_KEY,
    CANONICAL_WRIST_IMAGE_KEY,
    VLAProtocolAdapter,
    build_qwen_gr00t_serve_contract,
    build_request_protocol,
)
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.processor import DataProcessorPipeline
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
        self.server_metadata = {}

        self.load_model()

    def load_model(self):
        t_s = time.perf_counter()
        model_variant = self.config_engine.model_variant
        policy_cls = getattr(importlib.import_module("flagscale.models.vla"), model_variant)
        self.model, self.preprocessor, self.postprocessor = load_checkpoint(
            self.config_engine.model, policy_cls, self.config_engine.device
        )
        _backfill_model_input_features(self.model, self.preprocessor)

        protocol = build_request_protocol(self.config_engine.get("protocol"))
        rename_map = _build_rename_map(self.config_engine.get("rename_map"))
        _validate_protocol_mapping(rename_map, protocol)

        require_right_image = (
            protocol.right_image_key is not None and CANONICAL_RIGHT_IMAGE_KEY in rename_map
        )
        self.adapter = VLAProtocolAdapter(
            protocol=protocol,
            serve_contract=build_qwen_gr00t_serve_contract(
                task_key=self.config_engine.get("task_key"),
                require_right_image=require_right_image,
                image_hw=_resolve_config_image_hw(self.config_engine),
            ),
        )
        self.preprocessor = _load_preprocessor_with_overrides(
            checkpoint_dir=self.config_engine.model,
            rename_map=rename_map,
            device=str(self.config_engine.device),
        )
        self.server_metadata = {"env": self.adapter.protocol.env_name}

        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")
        logger.info(
            "Configured request protocol: "
            f"env={self.adapter.protocol.env_name}, "
            f"image={self.adapter.protocol.image_key}, "
            f"wrist={self.adapter.protocol.wrist_image_key}, "
            f"right={self.adapter.protocol.right_image_key}, "
            f"state={self.adapter.protocol.state_key}, "
            f"prompt={self.adapter.protocol.prompt_key}, "
            f"actions={self.adapter.protocol.actions_key}"
        )
        logger.info(
            "Configured QwenGr00t serving contract: "
            f"task={self.adapter.serve_contract.task_key}, "
            f"require_right_image={self.adapter.serve_contract.require_right_image}"
        )
        logger.info(f"Configured rename_map: {rename_map}")
        logger.info(f"Configured image size: {self.adapter.serve_contract.image_hw}")

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


def _build_rename_map(rename_map_cfg: Mapping[str, str] | None) -> dict[str, str]:
    if not rename_map_cfg:
        raise ValueError("Serving expects engine_args.rename_map to define observation remapping.")

    rename_map: dict[str, str] = {}
    for source_key, target_key in rename_map_cfg.items():
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("Serving expects rename_map keys to be non-empty strings.")
        if not isinstance(target_key, str) or not target_key:
            raise ValueError("Serving expects rename_map values to be non-empty strings.")
        rename_map[source_key] = target_key
    return rename_map


def _validate_protocol_mapping(rename_map: dict[str, str], protocol) -> None:
    required_sources = [CANONICAL_IMAGE_KEY, CANONICAL_WRIST_IMAGE_KEY, CANONICAL_STATE_KEY]
    missing_sources = [
        source_key for source_key in required_sources if source_key not in rename_map
    ]
    if missing_sources:
        raise ValueError(
            f"Serving rename_map is missing required observation mappings for {missing_sources}."
        )
    if protocol.prompt_key in rename_map:
        raise ValueError(
            "Serving rename_map only applies to observation keys. Configure prompt -> task with "
            "engine_args.task_key instead."
        )


def _load_preprocessor_with_overrides(
    *, checkpoint_dir: str, rename_map: dict[str, str], device: str
):
    pretrained_dir = f"{checkpoint_dir}/{PRETRAINED_MODEL_DIR}"
    return DataProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_dir,
        config_filename="policy_preprocessor.json",
        overrides={
            "rename_observations_processor": {"rename_map": rename_map},
            "device_processor": {"device": device},
        },
    )


def _backfill_model_input_features(model, preprocessor) -> None:
    if getattr(model, "input_features", None):
        return
    if preprocessor is None:
        return

    recovered_features = None
    for step in getattr(preprocessor, "steps", []):
        step_features = getattr(step, "features", None)
        if step_features:
            recovered_features = {
                key: feature
                for key, feature in step_features.items()
                if getattr(feature, "type", None) is not FeatureType.ACTION
            }
            if recovered_features:
                break

    if recovered_features:
        model.input_features = recovered_features
        logger.info(
            "Recovered model.input_features from checkpoint preprocessor for serving "
            f"({list(recovered_features.keys())})."
        )


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
        metadata=policy.server_metadata,
    )
    logger.info("Server running ...")
    server.serve_forever()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"][0])
