# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
LeRobot dataloader for Pi05 training aligned to OpenPI Libero pipeline.
"""

from __future__ import annotations

import inspect
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flagscale.models.pi05.tokenizer import DEFAULT_TOKENIZER_PATH, PaligemmaTokenizer

logger = logging.getLogger(__name__)


class DataTransformFn(Protocol):
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    def _apply(data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in transforms:
            data = transform(data)
        return data

    return _apply


def _pad_to_dim(array: np.ndarray, dim: int) -> np.ndarray:
    if array.shape[-1] >= dim:
        return array[..., :dim]
    pad_width = [(0, 0) for _ in range(array.ndim)]
    pad_width[-1] = (0, dim - array.shape[-1])
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def _parse_image(image: Any) -> np.ndarray:
    if isinstance(image, dict) and "bytes" in image:
        from PIL import Image

        image_bytes = image["bytes"]
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(pil)

    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return array


def _to_float_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
    return np.transpose(image, (2, 0, 1))


def _load_norm_stats(norm_stats_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = json.loads(norm_stats_path.read_text())
    stats = data.get("norm_stats", data)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in stats.items():
        out[key] = {
            "mean": np.asarray(value["mean"], dtype=np.float32),
            "std": np.asarray(value["std"], dtype=np.float32),
        }
        if "q01" in value and "q99" in value:
            out[key]["q01"] = np.asarray(value["q01"], dtype=np.float32)
            out[key]["q99"] = np.asarray(value["q99"], dtype=np.float32)
    return out


@dataclass(frozen=True)
class PromptFromLeRobotTask:
    tasks: Dict[int, str]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" in data:
            return data
        if "task" in data:
            return {**data, "prompt": data["task"]}
        if "language_instruction" in data:
            return {**data, "prompt": data["language_instruction"]}
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')
        task_index = int(data["task_index"])
        if task_index not in self.tasks:
            raise ValueError(f"{task_index=} not found in task mapping")
        return {**data, "prompt": self.tasks[task_index]}


@dataclass(frozen=True)
class LiberoInputs:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base_image = _parse_image(data.get("observation/image", data.get("image")))
        wrist_image = _parse_image(data.get("observation/wrist_image", data.get("wrist_image")))
        right_wrist_image = _parse_image(
            data.get(
                "observation/right_wrist_image",
                data.get("right_wrist_image", data.get("observation.images.image3")),
            )
        )

        def _ensure_hwc(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if image is None:
                return None
            array = np.asarray(image)
            if array.ndim != 3:
                return None
            return array

        base_image = _ensure_hwc(base_image)
        wrist_image = _ensure_hwc(wrist_image)
        right_wrist_image = _ensure_hwc(right_wrist_image)

        def _zeros_like_or_default(image: Optional[np.ndarray]) -> np.ndarray:
            if image is None:
                return np.zeros((224, 224, 3), dtype=np.uint8)
            return np.zeros_like(image)

        if base_image is None:
            base_image = wrist_image if wrist_image is not None else right_wrist_image
        if base_image is None:
            base_image = _zeros_like_or_default(None)

        if wrist_image is None:
            wrist_image = base_image if base_image is not None else right_wrist_image
        if wrist_image is None:
            wrist_image = _zeros_like_or_default(base_image)

        if right_wrist_image is None:
            # Mirror OpenPI's 3-camera expectation by duplicating the wrist view when missing.
            right_wrist_image = wrist_image if wrist_image is not None else base_image
        if right_wrist_image is None:
            right_wrist_image = _zeros_like_or_default(base_image)

        inputs: Dict[str, Any] = {
            "state": np.asarray(data.get("observation/state", data.get("state")), dtype=np.float32),
            "image": {
                "base_0_rgb": _to_float_chw(base_image),
                "left_wrist_0_rgb": _to_float_chw(wrist_image),
                "right_wrist_0_rgb": _to_float_chw(right_wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclass(frozen=True)
class Normalize:
    norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]]
    use_quantiles: bool = False

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.norm_stats:
            return data
        for key in ("state", "actions"):
            if key not in data or key not in self.norm_stats:
                continue
            stats = self.norm_stats[key]
            if self.use_quantiles and "q01" in stats and "q99" in stats:
                q01 = stats["q01"][..., : data[key].shape[-1]]
                q99 = stats["q99"][..., : data[key].shape[-1]]
                data[key] = (data[key] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean = stats["mean"][..., : data[key].shape[-1]]
                std = stats["std"][..., : data[key].shape[-1]]
                data[key] = (data[key] - mean) / (std + 1e-6)
        return data


@dataclass(frozen=True)
class TokenizePrompt:
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()

        state = data["state"] if self.discrete_state_input else None
        tokens, mask = self.tokenizer.tokenize(prompt, state)
        data["tokenized_prompt"] = tokens
        data["tokenized_prompt_mask"] = mask
        return data


@dataclass(frozen=True)
class PadStatesAndActions:
    model_action_dim: int

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["state"] = _pad_to_dim(data["state"], self.model_action_dim)
        if "actions" in data:
            data["actions"] = _pad_to_dim(data["actions"], self.model_action_dim)
        return data


class LeRobotDatasetWrapper(Dataset):
    def __init__(
        self,
        data_path: Path,
        *,
        action_horizon: int,
        action_dim: int,
        prompt_from_task: bool = True,
        tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
        tokenizer_max_len: int = 200,
        discrete_state_input: bool = False,
        use_quantile_norm: bool = False,
        norm_stats_path: Optional[Path] = None,
    ) -> None:
        self.data_path = data_path
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        dataset, tasks = _create_lerobot_dataset(data_path, action_horizon=action_horizon)
        self.dataset = dataset

        norm_stats = None
        if norm_stats_path is not None:
            norm_stats = _load_norm_stats(norm_stats_path)

        transforms: List[DataTransformFn] = []
        if prompt_from_task:
            transforms.append(PromptFromLeRobotTask(tasks))
        transforms += [
            LiberoInputs(),
            Normalize(norm_stats, use_quantiles=use_quantile_norm),
            TokenizePrompt(
                PaligemmaTokenizer(max_len=tokenizer_max_len, tokenizer_path=tokenizer_path),
                discrete_state_input=discrete_state_input,
            ),
            PadStatesAndActions(action_dim),
        ]

        self._transform = compose(transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        sample = self._transform(sample)

        actions = sample.get("actions")
        if actions is None:
            raise ValueError("Sample is missing actions")
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"actions horizon mismatch: expected {self.action_horizon}, got {actions.shape[0]}"
            )
        sample["actions"] = actions

        return sample


def _load_tasks_from_meta(dataset_path: Path) -> Dict[int, str]:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(tasks_path)
            if "task" in df and "task_index" in df:
                return {int(idx): task for idx, task in zip(df["task_index"], df["task"])}
        except Exception:
            pass
    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        try:
            tasks: Dict[int, str] = {}
            for line in tasks_jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if "task_index" in payload and "task" in payload:
                    tasks[int(payload["task_index"])] = payload["task"]
            if tasks:
                return tasks
        except Exception:
            pass
    return {}


def _create_lerobot_dataset(
    data_path: Path, *, action_horizon: int
) -> Tuple[Dataset, Dict[int, str]]:
    try:
        from lerobot.common.datasets import lerobot_dataset
    except Exception:  # pragma: no cover - dependency-driven
        return _create_lerobot_dataset_fallback(data_path, action_horizon=action_horizon)

    tasks = _load_tasks_from_meta(data_path)

    # Estimate fps from meta/info.json if available, otherwise fallback to 10.
    fps = 10
    info_path = data_path / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            fps = int(info.get("fps", fps))
        except Exception:
            pass

    delta_timestamps = {"actions": [t / fps for t in range(action_horizon)]}

    dataset_kwargs = {}
    dataset_sig = inspect.signature(lerobot_dataset.LeRobotDataset)
    if "delta_timestamps" in dataset_sig.parameters:
        dataset_kwargs["delta_timestamps"] = delta_timestamps
    if "local_dir" in dataset_sig.parameters:
        dataset_kwargs["local_dir"] = str(data_path)

    dataset = lerobot_dataset.LeRobotDataset(str(data_path), **dataset_kwargs)
    return dataset, tasks


def _create_lerobot_dataset_fallback(
    data_path: Path, *, action_horizon: int
) -> Tuple[Dataset, Dict[int, str]]:
    """Fallback loader that reads LeRobot parquet files via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required to load LeRobot parquet files when lerobot is unavailable."
        ) from exc

    parquet_files = sorted((data_path / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_path}/data")

    tasks = _load_tasks_from_meta(data_path)

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
    )

    return dataset, tasks


def _collate(items: List[Any]) -> Any:
    if isinstance(items[0], dict):
        return {k: _collate([i[k] for i in items]) for k in items[0]}
    if isinstance(items[0], np.ndarray):
        return torch.from_numpy(np.stack(items, axis=0))
    if isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0)
    return torch.tensor(items)


def create_pi05_dataloader(
    *,
    data_path: str,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    prompt_from_task: bool = True,
    tokenizer_path: Optional[str] = None,
    tokenizer_max_len: int = 200,
    discrete_state_input: bool = False,
    use_quantile_norm: bool = False,
    norm_stats_path: Optional[str] = None,
) -> DataLoader:
    dataset = LeRobotDatasetWrapper(
        Path(data_path),
        action_horizon=action_horizon,
        action_dim=action_dim,
        prompt_from_task=prompt_from_task,
        tokenizer_path=Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH,
        tokenizer_max_len=tokenizer_max_len,
        discrete_state_input=discrete_state_input,
        use_quantile_norm=use_quantile_norm,
        norm_stats_path=Path(norm_stats_path) if norm_stats_path else None,
    )

    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

    per_rank_batch_size = batch_size
    if sampler is not None:
        world_size = torch.distributed.get_world_size()
        per_rank_batch_size = batch_size // world_size
        if per_rank_batch_size < 1:
            message = (
                "Distributed batch_size must be >= world_size "
                f"(batch_size={batch_size}, world_size={world_size})."
            )
            logger.error(message)
            raise ValueError(message)

    return DataLoader(
        dataset,
        batch_size=per_rank_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate,
    )
