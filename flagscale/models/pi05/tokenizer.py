# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
Paligemma tokenizer utilities for Pi05.
This mirrors OpenPI tokenization behavior while staying local-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


DEFAULT_TOKENIZER_PATH = Path("/share/pi05_models/openpi05_base/paligemma_tokenizer.model")


@dataclass
class PaligemmaTokenizer:
    max_len: int = 200
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH

    def __post_init__(self) -> None:
        try:
            import sentencepiece as spm
        except Exception as exc:  # pragma: no cover - dependency-driven
            raise ImportError(
                "sentencepiece is required for PaligemmaTokenizer. "
                "Install it in the flagscale-train environment."
            ) from exc

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Paligemma tokenizer model not found: {self.tokenizer_path}")

        self._spm = spm.SentencePieceProcessor()
        self._spm.Load(str(self.tokenizer_path))

    def tokenize(
        self, prompt: str, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # Pi05 discrete state in prompt (OpenPI behavior).
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._spm.Encode(full_prompt, add_bos=True)
        else:
            # Pi0-style prompt for Pi05 with continuous state (OpenPI behavior).
            tokens = self._spm.Encode(cleaned_text, add_bos=True) + self._spm.Encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self.max_len:
            padding = [False] * (self.max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self.max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating. "
                    "Consider increasing max_token_len if this happens frequently.",
                    len(tokens),
                    self.max_len,
                )
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len

        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)
