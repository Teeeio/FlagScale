import numpy as np
import pytest
import torch

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION
from flagscale.serve.vla_starvla_adapter import (
    STARVLA_IMAGE_KEY,
    STARVLA_PROMPT_KEY,
    STARVLA_RIGHT_IMAGE_KEY,
    STARVLA_STATE_KEY,
    STARVLA_WRIST_IMAGE_KEY,
    StarVLASimAdapter,
    extract_policy_input_features,
    resolve_policy_image_hw,
    resolve_policy_input_layout,
)
from flagscale.train.processor.batch_processor import AddBatchDimensionProcessorStep
from flagscale.train.processor.converters import batch_to_transition, transition_to_batch


def _visual(shape=(256, 256, 3)):
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


def _state(shape=(8,)):
    return PolicyFeature(type=FeatureType.STATE, shape=shape)


def _action(shape=(7,)):
    return PolicyFeature(type=FeatureType.ACTION, shape=shape)


def test_adapter_remaps_starvla_keys_to_internal_features():
    features = {
        "observation.images.image": _visual(),
        "observation.images.wrist_image": _visual(),
        "observation.state": _state(),
    }
    layout = resolve_policy_input_layout(features, image_hw=(224, 224))
    adapter = StarVLASimAdapter(layout)

    obs = {
        STARVLA_IMAGE_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
        STARVLA_WRIST_IMAGE_KEY: np.ones((320, 320, 3), dtype=np.uint8),
        STARVLA_STATE_KEY: np.arange(8, dtype=np.float32),
        STARVLA_PROMPT_KEY: "PrepareCoffee",
    }

    batch = adapter.adapt_request(obs)

    assert sorted(batch) == [
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
        "task",
    ]
    assert isinstance(batch["observation.images.image"], torch.Tensor)
    assert isinstance(batch["observation.state"], torch.Tensor)
    assert batch["observation.images.image"].dtype == torch.uint8
    assert batch["observation.state"].dtype == torch.float32
    assert batch["observation.images.image"].shape == (224, 224, 3)
    assert batch["observation.images.wrist_image"].shape == (224, 224, 3)
    assert batch["observation.state"].shape == (8,)
    assert batch["task"] == "PrepareCoffee"

    batched = transition_to_batch(AddBatchDimensionProcessorStep()(batch_to_transition(batch)))
    assert batched["observation.images.image"].shape == (1, 224, 224, 3)
    assert batched["observation.state"].shape == (1, 8)
    assert batched["task"] == ["PrepareCoffee"]


def test_adapter_remaps_three_view_checkpoint_names():
    features = {
        "observation.images.base_0_rgb": _visual(),
        "observation.images.left_wrist_0_rgb": _visual(),
        "observation.images.right_wrist_0_rgb": _visual(),
        "observation.state": _state(),
    }
    layout = resolve_policy_input_layout(features, image_hw=(224, 224))
    adapter = StarVLASimAdapter(layout)

    obs = {
        STARVLA_IMAGE_KEY: np.full((100, 120, 3), 5, dtype=np.uint8),
        STARVLA_WRIST_IMAGE_KEY: np.full((100, 120, 3), 9, dtype=np.uint8),
        STARVLA_RIGHT_IMAGE_KEY: np.full((100, 120, 3), 17, dtype=np.uint8),
        STARVLA_STATE_KEY: np.arange(8, dtype=np.float32),
        STARVLA_PROMPT_KEY: "DrawTriangle",
    }
    batch = adapter.adapt_request(obs)

    assert batch["observation.images.base_0_rgb"][0, 0, 0] == 5
    assert batch["observation.images.left_wrist_0_rgb"][0, 0, 0] == 9
    assert batch["observation.images.right_wrist_0_rgb"][0, 0, 0] == 17


def test_adapter_errors_when_required_right_view_missing():
    features = {
        "observation.images.base_0_rgb": _visual(),
        "observation.images.left_wrist_0_rgb": _visual(),
        "observation.images.right_wrist_0_rgb": _visual(),
    }
    layout = resolve_policy_input_layout(features, image_hw=(224, 224))
    adapter = StarVLASimAdapter(layout, missing_image_policy="error")

    obs = {
        STARVLA_IMAGE_KEY: np.zeros((64, 64, 3), dtype=np.uint8),
        STARVLA_WRIST_IMAGE_KEY: np.zeros((64, 64, 3), dtype=np.uint8),
        STARVLA_PROMPT_KEY: "task",
    }

    with pytest.raises(ValueError, match="observation/image_right"):
        adapter.adapt_request(obs)


def test_adapter_can_fill_missing_image_with_black_frame():
    features = {
        "observation.images.base_0_rgb": _visual(),
        "observation.images.left_wrist_0_rgb": _visual(),
        "observation.images.right_wrist_0_rgb": _visual(),
    }
    layout = resolve_policy_input_layout(features, image_hw=(32, 48))
    adapter = StarVLASimAdapter(layout, missing_image_policy="black")

    obs = {
        STARVLA_IMAGE_KEY: np.zeros((64, 64, 3), dtype=np.uint8),
        STARVLA_WRIST_IMAGE_KEY: np.ones((64, 64, 3), dtype=np.uint8),
        STARVLA_PROMPT_KEY: "task",
    }
    batch = adapter.adapt_request(obs)

    assert batch["observation.images.right_wrist_0_rgb"].shape == (32, 48, 3)
    assert torch.count_nonzero(batch["observation.images.right_wrist_0_rgb"]) == 0


def test_format_response_returns_actions_list_of_lists():
    features = {"observation.images.image": _visual()}
    layout = resolve_policy_input_layout(features, image_hw=(224, 224))
    adapter = StarVLASimAdapter(layout)

    response = adapter.format_response({ACTION: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])})

    assert response == {"actions": [[1.0, 2.0], [3.0, 4.0]]}


def test_extract_input_features_falls_back_to_preprocessor_features():
    class _Model:
        input_features = {}

    class _Step:
        features = {
            "observation.images.image": _visual(),
            "observation.state": _state(),
            ACTION: _action(),
        }

    class _Preprocessor:
        steps = [_Step()]

    features = extract_policy_input_features(_Model(), _Preprocessor())
    assert sorted(features) == ["observation.images.image", "observation.state"]


def test_resolve_policy_image_hw_prefers_training_resolution():
    class _VLAData:
        default_image_resolution = [3, 224, 224]

    class _Data:
        vla_data = _VLAData()

    class _Config:
        data = _Data()

    image_hw = resolve_policy_image_hw(
        _Config(), {"observation.images.image": _visual((256, 256, 3))}
    )
    assert image_hw == (224, 224)
