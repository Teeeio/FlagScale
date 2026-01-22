"""Utilities for freezing model parameters using pattern matching."""

import re
from collections import defaultdict
from collections.abc import Generator, Iterable

import torch
import torch.nn as nn

from flagscale.runner.utils import logger

# TODO: (yupu) Add user-friendly interface for freezing backbones/heads/etc.


def freeze_and_get_trainable_params(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    freeze_patterns: list[str] | None = None,
    keep_patterns: list[str] | None = None,
) -> Generator[torch.nn.Parameter, None, None]:
    """
    Freeze parameters matching patterns and yield only trainable parameters.

    Args:
        named_parameters: Output of model.named_parameters()
        freeze_patterns: Regex patterns for params to freeze
        keep_patterns: Regex patterns for params to keep trainable (overrides `freeze_patterns`)

    Yields:
        Only parameters that should be trained (for optimizer).
    """
    freeze_patterns = freeze_patterns or []
    keep_patterns = keep_patterns or []

    compiled_freeze = [re.compile(p) for p in freeze_patterns]
    compiled_keep = [re.compile(p) for p in keep_patterns]
    freeze_counter = {p: 0 for p in freeze_patterns}
    keep_counter = {p: 0 for p in keep_patterns}

    def _should_freeze(name: str) -> bool:
        for i, pattern in enumerate(compiled_freeze):
            if pattern.search(name):
                freeze_counter[freeze_patterns[i]] += 1
                return True
        return False

    def _should_keep(name: str) -> bool:
        for i, pattern in enumerate(compiled_keep):
            if pattern.search(name):
                keep_counter[keep_patterns[i]] += 1
                return True
        return False

    trainable_count, frozen_count = 0, 0

    for name, param in named_parameters:
        should_freeze = _should_freeze(name) and not _should_keep(name)

        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()
            yield param

    total = trainable_count + frozen_count
    pct = trainable_count / total if total > 0 else 0
    logger.info(
        f"Parameters: trainable={trainable_count:,} ({pct:.2%}) | "
        f"frozen={frozen_count:,} | total={total:,}"
    )

    unused_freeze = [p for p, count in freeze_counter.items() if count == 0]
    if unused_freeze:
        logger.warning(f"Freeze patterns matched NOTHING (bad regex?): {unused_freeze}")

    unused_keep = [p for p, count in keep_counter.items() if count == 0]
    if unused_keep:
        logger.warning(f"Keep patterns matched NOTHING (bad regex?): {unused_keep}")


def apply_freeze_config(model: nn.Module, freeze_config) -> list:
    """
    Apply freeze config and return list of trainable parameters for optimizer.

    Args:
        model: The model to freeze
        freeze_config: FreezeConfig with freeze_patterns and keep_patterns

    Returns:
        List of trainable parameters (pass directly to optimizer)
    """
    if freeze_config is None:
        return list(model.parameters())

    return list(
        freeze_and_get_trainable_params(
            model.named_parameters(),
            freeze_patterns=freeze_config.freeze_patterns,
            keep_patterns=freeze_config.keep_patterns,
        )
    )


def log_trainable_params(model: nn.Module) -> dict:
    """Log trainable/frozen parameter statistics by module."""
    trainable_by_module = defaultdict(int)
    frozen_by_module = defaultdict(int)

    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if param.requires_grad:
            trainable_by_module[module_name] += param.numel()
        else:
            frozen_by_module[module_name] += param.numel()

    logger.info("=" * 60)
    logger.info("Parameter status by top-level module:")
    all_modules = set(trainable_by_module.keys()) | set(frozen_by_module.keys())
    for mod in sorted(all_modules):
        t = trainable_by_module.get(mod, 0)
        f = frozen_by_module.get(mod, 0)
        logger.info(f"  {mod}: {t:,} trainable, {f:,} frozen")
    logger.info("=" * 60)

    return {"trainable": dict(trainable_by_module), "frozen": dict(frozen_by_module)}


def print_param_names(model: nn.Module, pattern: str | None = None):
    """Debug helper: print parameter names (optionally filtered by pattern)."""
    for name, param in model.named_parameters():
        if pattern is None or re.search(pattern, name):
            status = "trainable" if param.requires_grad else "FROZEN"
            print(f"[{status}] {name}: {param.numel():,} params")
