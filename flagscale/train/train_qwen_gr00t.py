# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.optim import Optimizer

from flagscale.runner.utils import logger
from flagscale.train.train_config import TrainConfig, DataConfig
from flagscale.train.datasets.transforms import ImageTransforms
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.utils import dataset_to_policy_features
from flagscale.train.processor import PolicyProcessorPipeline
from flagscale.models.configs.types import PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_PREFIX, REWARD
from flagscale.models.configs.types import FeatureType
from flagscale.train.utils.logging_utils import (
    AverageMeter,
    MetricsTracker,
    format_big_number,
)
from flagscale.train.utils.train_utils import (
    save_checkpoint,
    get_step_checkpoint_dir,
    update_last_checkpoint,
)
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler
from flagscale.models.vla.qwen_gr00t import QwenGr00t
from flagscale.models.qwen_pi.qwen_pi import Qwen_PI

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from PIL import Image
from torch.utils.data import Dataset as TorchDataset


def collate_fn_starvla(batch):
    """Simple collate function that returns batch as list of dicts (starVLA style)."""
    return batch


class StarVLAFormatDataset(TorchDataset):
    """
    Wrapper dataset that converts FlagScale tensor images to match starVLA format.

    Conversion to match starVLA exactly:
    1. FlagScale tensor: float32 CHW, [0,1] range
    2. Convert to uint8 HWC: multiply by 255, permute, cast to uint8
    3. PIL.fromarray + resize (same as starVLA)

    starVLA format:
        dict(
            action=np.ndarray [T, action_dim],  # float16
            image=[PIL.Image, ...],             # list of PIL images (224x224)
            lang=str,                           # language instruction
        )
    """

    def __init__(
        self,
        dataset: "LeRobotDataset",
        image_keys: list[str] = None,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.dataset = dataset
        self.image_keys = image_keys or [
            "observation.images.image",
            "observation.images.wrist_image",
        ]
        self.image_size = image_size

        # Get action stats for min_max normalization (matching starVLA's StateActionTransform)
        action_stats = dataset.meta.stats.get("action", {})
        self.action_min = action_stats.get("min", None)
        self.action_max = action_stats.get("max", None)
        # Convert to numpy if needed
        if self.action_min is not None and hasattr(self.action_min, "numpy"):
            self.action_min = self.action_min.numpy()
        if self.action_max is not None and hasattr(self.action_max, "numpy"):
            self.action_max = self.action_max.numpy()

    def __len__(self):
        return len(self.dataset)

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    def _tensor_to_pil_starvla(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL exactly like starVLA:
        1. tensor is float32 CHW [0,1] from torchcodec
        2. Convert to uint8 HWC [0,255]
        3. PIL.fromarray + resize
        """
        # Remove batch dim if present
        if tensor.ndim == 4:
            tensor = tensor[0]

        # CHW -> HWC
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)

        # float32 [0,1] -> uint8 [0,255]
        img_np = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

        # PIL.fromarray + resize (exactly like starVLA)
        pil_img = Image.fromarray(img_np).resize(self.image_size)
        return pil_img

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Convert images to PIL format (matching starVLA processing)
        images = []
        for key in self.image_keys:
            if key in item:
                pil_img = self._tensor_to_pil_starvla(item[key])
                images.append(pil_img)

        # Get action (convert to numpy float16 like starVLA)
        action = item["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Apply min_max normalization (matching starVLA's Libero4in1DataConfig exactly)
        # starVLA only normalizes action.x, y, z, roll, pitch, yaw (indices 0-5)
        # action.gripper (index 6) is NOT normalized
        # Formula: 2 * (x - min) / (max - min) - 1
        if self.action_min is not None and self.action_max is not None:
            # Only normalize first 6 dimensions (x, y, z, roll, pitch, yaw)
            # Keep gripper (dim 6) as raw value
            normalize_dims = 6  # Only normalize first 6 dims
            action_range = self.action_max[:normalize_dims] - self.action_min[:normalize_dims]
            mask = action_range > 1e-8

            normalized = action.copy()
            # Normalize dimensions 0-5 where range > 0
            for i in range(normalize_dims):
                if mask[i]:
                    normalized[..., i] = (action[..., i] - self.action_min[i]) / action_range[i]
                    normalized[..., i] = 2.0 * normalized[..., i] - 1.0
                else:
                    normalized[..., i] = 0.0
            # Keep dimension 6 (gripper) as-is (no normalization)
            action = normalized

        action = action.astype(np.float16)

        # Get language instruction
        lang = item.get("task", "")
        if isinstance(lang, torch.Tensor):
            lang = lang.item() if lang.numel() == 1 else str(lang.tolist())

        # Get trajectory_id and frame_index for debugging (matching starVLA format)
        trajectory_id = item.get("episode_index", -1)
        if isinstance(trajectory_id, torch.Tensor):
            trajectory_id = trajectory_id.item()
        frame_index = item.get("index", idx)
        if isinstance(frame_index, torch.Tensor):
            frame_index = frame_index.item()

        return dict(
            action=action,
            image=images,
            lang=lang,
            trajectory_id=trajectory_id,
            frame_index=frame_index,
        )


def register_debug_hooks(model_obj):
    """
    给模型挂载带有 Rank 信息的 Forward 和 Backward Hook
    model_obj: 可以是 model (list) 也可以是 model[0] (module)
    """

    # 1. 获取 Rank 的辅助函数
    def get_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    # 2. 通用打印函数
    def calc_and_print(tensor, name, tag):
        """
        tensor: 要打印的张量
        name: 模块名称 + 参数位置
        tag: FWD 或 BWD
        """
        if tensor is None:
            return
        # 仅处理 Tensor，忽略 None 或其他类型
        if isinstance(tensor, torch.Tensor):
            # 获取当前 Rank
            rank = get_rank()
            # 计算 sum (转为 float32 防止溢出，item() 会触发同步确保数值准确)
            # 注意：打印日志会显著降低训练速度，仅用于 Debug
            val = torch.sum(tensor.detach().to(torch.float32)).item()
            # 打印格式：[Rank 0][FWD] layers.0.self_attention sum: 1234.56
            print(f"[Rank {rank}][{tag}] {name} sum: {val}", flush=True)

    # 3. 前向 Hook 定义
    def forward_wrapper(name):
        def forward_hook(module, input, output):
            # 打印 Input (元组或张量)
            if isinstance(input, (list, tuple)):
                for i, item in enumerate(input):
                    calc_and_print(item, f"{name}.input[{i}]", "FWD")
            else:
                calc_and_print(input, f"{name}.input", "FWD")
            # 打印 Output
            if isinstance(output, (list, tuple)):
                for i, item in enumerate(output):
                    calc_and_print(item, f"{name}.output[{i}]", "FWD")
            else:
                calc_and_print(output, f"{name}.output", "FWD")

        return forward_hook

    # 4. 反向 Hook 定义 (使用 register_full_backward_hook)
    def backward_wrapper(name):
        def backward_hook(module, grad_input, grad_output):
            # grad_output: 从上一层流回来的梯度 (反向传播的“输入”)
            if isinstance(grad_output, (list, tuple)):
                for i, g in enumerate(grad_output):
                    calc_and_print(g, f"{name}.grad_output[{i}]", "BWD")
            else:
                calc_and_print(grad_output, f"{name}.grad_output", "BWD")
            # grad_input: 当前层计算出的梯度 (准备传给下一层)
            if isinstance(grad_input, (list, tuple)):
                for i, g in enumerate(grad_input):
                    calc_and_print(g, f"{name}.grad_input[{i}]", "BWD")
            else:
                calc_and_print(grad_input, f"{name}.grad_input", "BWD")

        return backward_hook

    # 5. 开始注册
    # 兼容 list 结构
    actual_module = model_obj[0] if isinstance(model_obj, list) else model_obj
    print(f"Rank {get_rank()}: 开始挂载 Debug Hooks (仅叶子层)...", flush=True)
    # 遍历所有子模块
    for name, module in actual_module.named_modules():
        # 【核心修改】跳过容器层，只Hook叶子层（没有子模块的层）
        # 这样可以避免 Hook 顶层模块导致的 View 属性变化，同时也能覆盖所有计算
        if len(list(module.children())) > 0:
            continue
        # 额外的黑名单（可选）：跳过一些不重要的层，比如 Dropout
        if isinstance(module, torch.nn.Dropout):
            continue
        # 注册 FWD Hook
        handle_fwd = module.register_forward_hook(forward_wrapper(name))
        # 注册 BWD Hook
        handle_bwd = module.register_full_backward_hook(backward_wrapper(name))


def remove_debug_hooks_force(model_obj):
    """
    暴力清除模型中所有的 hook，不需要 handle。
    """
    actual_module = model_obj[0] if isinstance(model_obj, list) else model_obj
    print("Force removing all hooks...", flush=True)
    for module in actual_module.modules():
        # 清除前向 hook
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        # 清除反向 hook
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
    print("Hooks force removed.", flush=True)




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False


def apply_fsdp2(policy, device_mesh):
    """Apply FSDP2 sharding to QwenGr00t.

    Uses a MixedPrecisionPolicy that matches DeepSpeed bf16 behavior:
      bf16.enabled=true + ZeRO-2 → param_dtype=bf16, reduce_dtype=bf16, reshard=False
    """
    # Cast everything to fp32 first so the root param group has uniform dtype.
    policy = policy.float()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}

    vlm_model = policy.vlm.model  # Qwen3VLForConditionalGeneration

    # reshard_after_forward=False keeps params unsharded during forward+backward
    reshard = False

    for block in vlm_model.model.visual.blocks:
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard)

    for layer in vlm_model.model.language_model.layers:
        fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)

    dit = policy.action_model._head.model
    for block in dit.transformer_blocks:
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard)

    fully_shard(policy, **fsdp_config)


def make_dataset(cfg: DataConfig):
    # TODO: (yupu) Support image transforms
    enable_image_transform = False
    # TODO: (yupu) Remove hard-coded video backend
    # After not much testing, It feels like that `torchcodec` is more robust than `pyav`
    # `pyav` crashes sometimes
    video_backend = "torchcodec"
    # video_backend = "torchvision_av"
    # video_backend = "pyav"

    # image_transforms = ImageTransforms(cfg.image_transforms) if enable_image_transform else None

    # Match starVLA: resize uint8 via PIL, then normalize to [0,1]
    def _resize_like_starvla(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor):
            return frames
        is_single = False
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
            is_single = True
        if frames.dim() != 4:
            return frames
        from PIL import Image
        import numpy as np

        resized_frames = []
        for frame in frames:
            channel_last = frame.shape[-1] in (1, 3, 4)
            if channel_last:
                frame_hwc = frame
            elif frame.shape[0] in (1, 3, 4):
                frame_hwc = frame.permute(1, 2, 0)
            else:
                frame_hwc = frame
                channel_last = True
            frame_uint8 = (frame_hwc * 255).round().clamp(0, 255).to(torch.uint8)
            pil = Image.fromarray(frame_uint8.cpu().numpy()).resize(
                (224, 224), resample=Image.BILINEAR
            )
            out = torch.from_numpy(np.array(pil)).to(frames.device).float() / 255.0
            if not channel_last:
                out = out.permute(2, 0, 1)
            resized_frames.append(out)
        output = torch.stack(resized_frames, dim=0)
        return output[0] if is_single else output

    def _resize_to_uint8_hwc(frame: torch.Tensor) -> torch.Tensor:
        """float32 CHW [0,1] from torchcodec → uint8 HWC 224x224 via PIL resize."""
        from PIL import Image
        import numpy as np

        frame_uint8 = (frame.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
        # PIL default is BICUBIC, matching starVLA's Image.fromarray(image).resize((224, 224))
        pil = Image.fromarray(frame_uint8.cpu().numpy()).resize((224, 224))
        return torch.from_numpy(np.array(pil))

    image_transforms = _resize_to_uint8_hwc
    # Leave the revision to None
    ds_meta = LeRobotDatasetMetadata(root=cfg.data_path, revision=None)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)

    dataset = LeRobotDataset(
        root=cfg.data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def resolve_delta_timestamps(
    cfg: DataConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg: The policy config (PI0Config or PI05Config) to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


# datasets/utils.py
def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.

    This is an equivalent of `itertools.cycle` but is safe for use with
    PyTorch DataLoaders with multiple workers.
    See https://github.com/pytorch/pytorch/issues/23900 for details.

    Args:
        iterable: The iterable to cycle over.

    Yields:
        Items from the iterable, restarting from the beginning when exhausted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def format_train_tracker_step(train_tracker: MetricsTracker) -> str:
    def _format_meter_val(meter: AverageMeter) -> str:
        fmt = meter.fmt[1:] if meter.fmt.startswith(":") else meter.fmt
        return f"{meter.name}:{format(meter.val, fmt)}"

    display_list = [
        f"step:{format_big_number(train_tracker.steps)}",
        f"smpl:{format_big_number(train_tracker.samples)}",
        f"ep:{format_big_number(train_tracker.episodes)}",
        f"epch:{train_tracker.epochs:.2f}",
        *[_format_meter_val(m) for m in train_tracker.metrics.values()],
    ]
    return " ".join(display_list)


# def validate_visual_features_consistency(
#     cfg: PI0Config,
#     features: dict[str, PolicyFeature],
# ) -> None:
#     """
#     Validates visual feature consistency between a policy config and provided dataset/environment features.

#     Args:
#         cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
#         features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
#     """
#     expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
#     provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
#     if not provided_visuals.issubset(expected_visuals):
#         raise_feature_mismatch_error(provided_visuals, expected_visuals)


def make_policy(
    config: TrainConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
):
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created (PI0Config or PI05Config).
             If `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).
        model_variant: Model variant to use, either "pi0" or "pi0.5".

    Returns:
        An instantiated and device-placed policy model (PI0Policy or PI05Policy).
    """

    # # Select policy class based on model variant
    # if model_variant == "pi0.5":
    #     policy_cls = PI05Policy
    # else:
    #     policy_cls = PI0Policy

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)

    # FIXME
    output_features = {
        # Changed from ft.type is FeatureType.ACTION to ft.type == FeatureType.ACTION
        # for different enum classes: flagscale.FeatureType vs lerobot.FeatureType
        key: ft
        for key, ft in features.items()
        if ft.type == FeatureType.ACTION
    }
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    # kwargs["config"] = config.model

    # PI0 finetuning, so always load a pretrained policy.
    # Load a pretrained policy and override the config if needed (for example, if there are inference-time
    # hyperparameters that we want to vary).
    # kwargs["pretrained_name_or_path"] = cfg.pretrained_path
    # policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg)

    # TODO: (yupu) This is a hack, we should find a better way to handle this. LeRobot does this in the policy config.
    # The order of the images is defined in the dataset config.json
    image_features = {
        key: ft for key, ft in input_features.items() if ft.type is FeatureType.VISUAL
    }
    config.data.vla_data.image_features = image_features

    policy = QwenGr00t(config=config)
    policy.to("cuda")

    return policy, input_features, output_features


def make_preprocessor_from_config(
    config: dict[str, Any] | list[str | dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """
    Create a preprocessor pipeline from step configurations with optional overrides.

    This function creates a PolicyProcessorPipeline directly from step configurations,
    without requiring a pretrained path. It supports overriding step configurations
    similar to PolicyProcessorPipeline.from_pretrained().

    Args:
        config: Can be either:
            - A dict with "name" and "steps" fields (JSON format):
              {"name": "policy_preprocessor", "steps": [...]}
            - A list of step configurations (concise format):
              ["step_name", {"step_name": {...}}]
        overrides: Optional dictionary to override step configurations. Keys should
            match the step's registry_name. Example:
            {"device_processor": {"device": "cuda"},
             "normalizer_processor": {"stats": dataset.meta.stats}}

    Returns:
        A PolicyProcessorPipeline instance with the configured steps.

    Example (JSON format with overrides):
        ```python
        config = {
            "name": "policy_preprocessor",
            "steps": [
                {"registry_name": "device_processor", "config": {"device": "cpu"}},
                {"registry_name": "normalizer_processor", "config": {"eps": 1e-8}},
            ],
        }
        overrides = {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": dataset.meta.stats, "features": {...}},
        }
        preprocessor = make_preprocessor_from_config(config, overrides=overrides)
        # device_processor will use device="cuda" (overridden)
        # normalizer_processor will use eps=1e-8 (from config) and stats from overrides
        ```

    Example (concise list format):
        ```python
        steps = [
            "rename_observations_processor",
            "device_processor",
            {"normalizer_processor": {"eps": 1e-8}},
        ]
        preprocessor = make_preprocessor_from_config(steps)
        ```

    Raises:
        ValueError: If a step configuration is invalid or step cannot be instantiated.
        KeyError: If a registry name is not found.
    """
    from flagscale.train.processor.pipeline import ProcessorStepRegistry

    overrides = overrides or {}

    # Determine format and extract step configs
    if isinstance(config, (dict, DictConfig)) and "steps" in config:
        # JSON format: {"name": "...", "steps": [...]}
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        step_configs = config["steps"]
        pipeline_name = config.get("name", "policy_preprocessor")
    elif isinstance(config, list):
        # Concise list format
        step_configs = config
        pipeline_name = "policy_preprocessor"
    else:
        raise ValueError(f"Config must be a dict with 'steps' key or a list, got {type(config)}")

    steps = []
    for step_entry in step_configs:
        # Determine step format and normalize to standard dict
        if isinstance(step_entry, str):
            # Concise format: "step_name"
            step_dict = {"registry_name": step_entry, "config": {}}
        elif isinstance(step_entry, (dict, DictConfig)):
            if "registry_name" in step_entry:
                # JSON format: {"registry_name": "...", "config": {...}}
                if isinstance(step_entry, DictConfig):
                    step_entry = OmegaConf.to_container(step_entry, resolve=True)
                step_dict = step_entry
            elif len(step_entry) == 1:
                # Concise format: {"step_name": {...}}
                step_name = next(iter(step_entry.keys()))
                step_config = step_entry[step_name]
                if isinstance(step_config, DictConfig):
                    step_config = OmegaConf.to_container(step_config, resolve=True)
                step_dict = {"registry_name": step_name, "config": step_config}
            else:
                raise ValueError(
                    f"Step config dict must have either 'registry_name' or exactly one key, "
                    f"got {list(step_entry.keys())}"
                )
        else:
            raise ValueError(
                f"Step config must be str or dict, got {type(step_entry)}: {step_entry}"
            )

        # Get step class
        registry_name = step_dict["registry_name"]
        step_class = ProcessorStepRegistry.get(registry_name)

        # Merge config with overrides (overrides take precedence)
        try:
            base_config = step_dict.get("config", {})
            step_overrides = overrides.get(registry_name, {})
            merged_config = {**base_config, **step_overrides}

            step_instance = step_class(**merged_config)
            steps.append(step_instance)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate processor step '{registry_name}' "
                f"with config {merged_config}. Error: {e!s}"
            ) from e

    return PolicyProcessorPipeline(
        steps=steps,
        name=pipeline_name,
    )


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    use_amp: bool,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
) -> MetricsTracker:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained (FSDP2-sharded).
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        use_amp: Whether to use automatic mixed precision.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        The updated MetricsTracker with new statistics for this step.
    """
    start_time = time.perf_counter()

    optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )
    with autocast_context:
        loss = policy(batch)

    loss.backward()

    # Clip gradients (torch.nn.utils.clip_grad_norm_ works with DTensors in PyTorch ≥2.6)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
    )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, 'full_tensor') else grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    return train_metrics


def main(config: TrainConfig, seed: int):
    set_seed(seed)

    # --- Distributed init ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = rank == 0

    dataset = make_dataset(config.data)
    dist.barrier()

    policy, input_features, output_features = make_policy(config=config, ds_meta=dataset.meta)
    dist.barrier()

    # --- Apply FSDP2 ---
    device_mesh = init_device_mesh("cuda", (world_size,))
    apply_fsdp2(policy, device_mesh)

    # Create processors - only provide dataset_stats if not resuming from saved processors
    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {
                **input_features,
                **output_features,
            },
        },
    }

    num_workers = 0  # config.system.num_workers
    shuffle = config.system.shuffle

    # DistributedSampler ensures each rank gets different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=config.system.batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Setup preprocessor
    preprocessor = None
    if config.data.preprocessor is not None:
        preprocessor = make_preprocessor_from_config(
            config.data.preprocessor, overrides=preprocessor_overrides
        )

    # Setup postprocessor (unnormalization for inference)
    postprocessor = None
    postprocessor_config = getattr(config.data, "postprocessor", None)
    if postprocessor_config is not None:
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **input_features,
                    **output_features,
                },
            },
        }
        postprocessor = make_preprocessor_from_config(
            postprocessor_config, overrides=postprocessor_overrides
        )

    # Setup optimizer and scheduler (applies freeze config internally)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(policy, config)

    dist.barrier()

    dl_iter = cycle(dataloader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = config.system.batch_size * world_size

    step = 0

    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    # Ensures proper data shuffling across epochs in distributed training
    epoch = 0
    samples_per_epoch = len(dataset) // effective_batch_size
    sampler.set_epoch(epoch)

    for _ in range(step, config.system.train_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        if preprocessor is not None:
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            use_amp=config.system.use_amp,
            grad_clip_norm=config.system.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()

        # Update epoch counter for sampler.set_epoch() when we've processed one epoch worth of samples
        # This ensures proper data shuffling across epochs in distributed training
        if samples_per_epoch > 0 and step % samples_per_epoch == 0:
            epoch += 1
            sampler.set_epoch(epoch)

        if step % config.system.log_freq == 0 and is_main_process:
            logger.info(f"step: {step} {format_train_tracker_step(train_tracker)}")
            train_tracker.reset_averages()

        if (
            config.system.checkpoint.save_checkpoint
            and step % config.system.checkpoint.save_freq == 0
        ):
            dist.barrier()

            # get_model_state_dict is a collective — all ranks must call it
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(policy, options=options)

            if is_main_process:
                from pathlib import Path

                logger.info(f"Saving checkpoint at step {step}")
                output_dir = Path(config.system.checkpoint.output_directory)
                checkpoint_dir = get_step_checkpoint_dir(
                    output_dir, config.system.train_steps, step
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    policy=state_dict,
                    config=config,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)

            dist.barrier()

    if is_main_process:
        logger.info("Training completed")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QwenGr00t model. This script is typically called by the flagscale runner, not directly."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config_file_path = args.config_file

    # Load config from YAML file (Hydra-generated config.yaml contains both train and experiment)
    config = OmegaConf.load(config_file_path)

    logger.info(f"full config: {config}")

    # Extract train config and convert to Pydantic TrainConfig (preserves raw configs)
    train_config = TrainConfig.from_hydra_config(config)

    # Extract experiment config (seed, exp_dir, etc.)
    experiment_config = OmegaConf.to_container(config.experiment, resolve=True)
    seed = experiment_config.get("seed", 42)

    logger.info("=" * 100)
    logger.info(f"Experiment: {experiment_config}")
    logger.info(f"Train config: {train_config}")

    main(train_config, seed)
