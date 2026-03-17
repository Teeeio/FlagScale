import asyncio
import socket
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import websockets
from omegaconf import OmegaConf

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION
from flagscale.serve import msgpack_numpy
from flagscale.serve.run_serve_vla import Policy
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer


def _visual(shape=(32, 48, 3)):
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


def _state(shape=(8,)):
    return PolicyFeature(type=FeatureType.STATE, shape=shape)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until_server_ready(host: str, port: int, *, timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            async with websockets.connect(f"ws://{host}:{port}", proxy=None) as websocket:
                await websocket.recv()
                return
        except (OSError, websockets.exceptions.WebSocketException):
            if loop.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for websocket server on {host}:{port}.")
            await asyncio.sleep(0.05)


class _FakePreprocessor:
    class _Step:
        features = {
            "observation.images.image": _visual(),
            "observation.images.wrist_image": _visual(),
            "observation.state": _state(),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        }

    steps = [_Step()]

    def __call__(self, batch):
        processed = dict(batch)
        for key, value in list(processed.items()):
            if isinstance(value, torch.Tensor) and value.ndim in (1, 3):
                processed[key] = value.unsqueeze(0)
        if isinstance(processed.get("task"), str):
            processed["task"] = [processed["task"]]
        return processed


class _FakePostprocessor:
    def __call__(self, batch):
        processed = dict(batch)
        processed[ACTION] = processed[ACTION] + 0.5
        return processed


# Mirror the QwenGr00t serving contract: batched tensors in, ACTION dict out.
class _FakeModel:
    def __init__(self):
        self.input_features = {
            "observation.images.image": _visual(),
            "observation.images.wrist_image": _visual(),
            "observation.state": _state(),
        }
        self._config = OmegaConf.create(
            {"data": {"vla_data": {"default_image_resolution": [3, 32, 48]}}}
        )
        self.last_batch = None

    def predict_action(self, batch):
        self.last_batch = batch
        assert batch["observation.images.image"].shape == (1, 32, 48, 3)
        assert batch["observation.images.wrist_image"].shape == (1, 32, 48, 3)
        assert batch["observation.state"].shape == (1, 8)
        assert batch["task"] == ["SmokeTask"]
        return {ACTION: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)}


class _FakeModelWithoutInputFeatures(_FakeModel):
    def __init__(self):
        super().__init__()
        self.input_features = {}


@pytest.mark.asyncio
async def test_websocket_policy_server_smoke_with_starvla_payload():
    host = "127.0.0.1"
    port = _unused_port()
    model = _FakeModel()
    config = OmegaConf.create(
        {
            "engine_args": {
                "host": host,
                "port": port,
                "model_variant": "FakeVariant",
                "model": "/tmp/fake-checkpoint",
                "device": "cpu",
                "missing_image_policy": "error",
            }
        }
    )

    with (
        patch(
            "flagscale.serve.run_serve_vla.load_checkpoint",
            return_value=(model, _FakePreprocessor(), _FakePostprocessor()),
        ),
        patch(
            "flagscale.serve.run_serve_vla.importlib.import_module",
            return_value=SimpleNamespace(FakeVariant=object),
        ),
    ):
        policy = Policy(config)

    server = WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata={"env": "starvla_sim"},
    )
    server_task = asyncio.create_task(server.run())
    await _wait_until_server_ready(host, port)

    try:
        async with websockets.connect(f"ws://{host}:{port}", proxy=None) as websocket:
            metadata = msgpack_numpy.unpackb(await websocket.recv())
            assert metadata == {"env": "starvla_sim"}

            observation = {
                "observation/image": np.zeros((80, 96, 3), dtype=np.uint8),
                "observation/wrist_image": np.ones((80, 96, 3), dtype=np.uint8),
                "observation/state": np.arange(8, dtype=np.float32),
                "prompt": "SmokeTask",
            }
            await websocket.send(msgpack_numpy.packb(observation))
            response = msgpack_numpy.unpackb(await websocket.recv())
    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task

    assert model.last_batch is not None
    assert response["actions"] == [[1.5, 2.5], [3.5, 4.5]]
    assert "server_timing" in response
    assert "infer_ms" in response["server_timing"]


def test_policy_backfills_model_input_features_from_preprocessor():
    config = OmegaConf.create(
        {
            "engine_args": {
                "host": "127.0.0.1",
                "port": 5000,
                "model_variant": "FakeVariant",
                "model": "/tmp/fake-checkpoint",
                "device": "cpu",
                "missing_image_policy": "error",
            }
        }
    )
    model = _FakeModelWithoutInputFeatures()
    preprocessor = _FakePreprocessor()

    with (
        patch(
            "flagscale.serve.run_serve_vla.load_checkpoint",
            return_value=(model, preprocessor, _FakePostprocessor()),
        ),
        patch(
            "flagscale.serve.run_serve_vla.importlib.import_module",
            return_value=SimpleNamespace(FakeVariant=object),
        ),
    ):
        policy = Policy(config)

    assert sorted(policy.model.input_features) == [
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
    ]
