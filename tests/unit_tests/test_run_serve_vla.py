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

from flagscale.models.utils.constants import ACTION
from flagscale.serve import msgpack_numpy
from flagscale.serve.run_serve_vla import Policy
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    RenameObservationsProcessorStep,
)
from flagscale.train.processor.batch_processor import AddBatchDimensionProcessorStep


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


class _FakePostprocessor:
    def __call__(self, batch):
        processed = dict(batch)
        processed[ACTION] = processed[ACTION] + 0.5
        return processed


# Mirror the QwenGr00t serving contract: internal observation keys + batched tensors in, ACTION dict out.
class _FakeModel:
    def __init__(self):
        self.last_batch = None

    def predict_action(self, batch):
        self.last_batch = batch
        assert {
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            "task",
        }.issubset(batch)
        assert batch["observation.images.image"].shape == (1, 32, 48, 3)
        assert batch["observation.images.wrist_image"].shape == (1, 32, 48, 3)
        assert batch["observation.state"].shape == (1, 8)
        assert batch["task"] == ["SmokeTask"]
        return {ACTION: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)}


def _base_preprocessor():
    # The saved checkpoint preprocessor already contains rename/to_batch/device in QwenGr00t training.
    return DataProcessorPipeline(
        steps=[AddBatchDimensionProcessorStep()],
        name="PolicyPreprocessor",
    )


def _preprocessor_from_overrides(*args, **kwargs):
    overrides = kwargs["overrides"]
    return DataProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(
                rename_map=overrides["rename_observations_processor"]["rename_map"]
            ),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=overrides["device_processor"]["device"]),
        ],
        name="PolicyPreprocessor",
    )


def _policy_config(host: str, port: int):
    return OmegaConf.create(
        {
            "engine_args": {
                "host": host,
                "port": port,
                "model_variant": "FakeVariant",
                "model": "/tmp/fake-checkpoint",
                "device": "cpu",
                "image_hw": [32, 48],
                "protocol": {
                    "env_name": "test_protocol",
                    "image_key": "observation/image",
                    "wrist_image_key": "observation/wrist_image",
                    "state_key": "observation/state",
                    "prompt_key": "prompt",
                    "actions_key": "actions",
                },
                "rename_map": {
                    "observation.image": "observation.images.image",
                    "observation.wrist_image": "observation.images.wrist_image",
                    "observation.state": "observation.state",
                },
                "task_key": "task",
            }
        }
    )


@pytest.mark.asyncio
async def test_websocket_policy_server_smoke_with_protocol_payload():
    host = "127.0.0.1"
    port = _unused_port()
    model = _FakeModel()
    config = _policy_config(host, port)

    with (
        patch(
            "flagscale.serve.run_serve_vla.load_checkpoint",
            return_value=(model, _base_preprocessor(), _FakePostprocessor()),
        ),
        patch(
            "flagscale.serve.run_serve_vla.DataProcessorPipeline.from_pretrained",
            side_effect=_preprocessor_from_overrides,
        ) as mocked_preprocessor,
        patch(
            "flagscale.serve.run_serve_vla.importlib.import_module",
            return_value=SimpleNamespace(FakeVariant=object),
        ),
    ):
        policy = Policy(config)

    step_names = [
        getattr(step.__class__, "_registry_name", step.__class__.__name__)
        for step in policy.preprocessor.steps
    ]
    assert step_names == ["rename_observations_processor", "to_batch_processor", "device_processor"]
    assert mocked_preprocessor.call_args.kwargs["overrides"] == {
        "rename_observations_processor": {
            "rename_map": {
                "observation.image": "observation.images.image",
                "observation.wrist_image": "observation.images.wrist_image",
                "observation.state": "observation.state",
            }
        },
        "device_processor": {"device": "cpu"},
    }

    server = WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy.server_metadata,
    )
    server_task = asyncio.create_task(server.run())
    await _wait_until_server_ready(host, port)

    try:
        async with websockets.connect(f"ws://{host}:{port}", proxy=None) as websocket:
            metadata = msgpack_numpy.unpackb(await websocket.recv())
            assert metadata == {"env": "test_protocol"}

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
