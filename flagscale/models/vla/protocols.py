from typing import Protocol

from torch import Tensor


class VLMBackbone(Protocol):
    @property
    def config(self):
        """HF config object (e.g., Qwen2VLConfig)."""
        ...

    def prepare_input(self, batch: dict) -> dict[str, Tensor]:
        """
        Args:
            batch: Raw batch with 'image', 'lang', etc.
        Returns:
            Tokenized inputs ready for forward().
        """
        ...

    def forward(self, batch: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """
        Args:
            batch: Tokenized inputs from prepare_input().
        Returns:
            dict with 'hidden_states': tuple of layer outputs.
        """
        ...


# TODO: (yupu) This `ActionModel` assumes that the VLA model is a composite of a VLM and an ActionModel.
class ActionModel(Protocol):
    def forward(
        self, vlm_output: dict[str, Tensor], action_input: dict[str, Tensor], **kwargs
    ) -> dict[str, Tensor]:
        """
        Args:
            vlm_output: From VLM, contains 'hidden_states'.
            action_input: Raw batch - pick what you need ('actions', 'state', etc.).
        Returns:
            dict with 'loss'.
        """
        ...

    def predict(
        self, vlm_output: dict[str, Tensor], action_input: dict[str, Tensor], **kwargs
    ) -> dict[str, Tensor]:
        """
        Args:
            vlm_output: From VLM, contains 'hidden_states'.
            action_input: Raw batch - pick what you need ('state', etc.).
        Returns:
            dict with 'actions': Tensor [B, horizon, action_dim].
        """
        ...
