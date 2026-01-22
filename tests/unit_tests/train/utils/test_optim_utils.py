"""Unit tests for pattern-based module freezing utilities."""

import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

from flagscale.train.utils.optim_utils import (
    apply_freeze_config,
    freeze_and_get_trainable_params,
    log_trainable_params,
    print_param_names,
)


class SimpleModel(nn.Module):
    """Simple model for testing freeze patterns."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.head = nn.Linear(10, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class NestedModel(nn.Module):
    """Model with nested structure similar to QwenGR00T."""

    def __init__(self):
        super().__init__()
        self.vlm = nn.ModuleDict(
            {
                "visual": nn.Sequential(
                    nn.Linear(10, 20),
                    nn.Linear(20, 10),
                ),
                "language": nn.ModuleDict(
                    {
                        "layers": nn.ModuleList([nn.Linear(10, 10) for _ in range(5)]),
                        "embed": nn.Embedding(100, 10),
                    }
                ),
            }
        )
        self.action_model = nn.ModuleDict(
            {
                "encoder": nn.Linear(10, 20),
                "decoder": nn.Linear(20, 10),
                "transformer_blocks": nn.ModuleList([nn.Linear(10, 10) for _ in range(4)]),
            }
        )

    def forward(self, x):
        return x


class TestFreezeAndGetTrainableParams(unittest.TestCase):
    """Test freeze_and_get_trainable_params function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_no_patterns_all_trainable(self):
        """Without patterns, all params should be trainable."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=None,
                keep_patterns=None,
            )
        )

        all_params = list(self.model.parameters())
        self.assertEqual(len(params), len(all_params))

        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    def test_freeze_single_module(self):
        """Test freezing a single module by pattern."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        # Check encoder is frozen
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should only be trainable ones
        for param in params:
            self.assertTrue(param.requires_grad)

    def test_freeze_multiple_modules(self):
        """Test freezing multiple modules."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*", "decoder\\..*"],
                keep_patterns=None,
            )
        )

        # Only head should be trainable
        for name, param in self.model.named_parameters():
            if name.startswith("head"):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

        # Returned params should only be head params
        head_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("head")
        )
        self.assertEqual(len(params), head_param_count)

    def test_freeze_all_pattern(self):
        """Test freezing everything with '.*' pattern."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=[".*"],
                keep_patterns=None,
            )
        )

        self.assertEqual(len(params), 0)
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_keep_patterns_override_freeze(self):
        """Test that keep_patterns override freeze_patterns."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=[".*"],  # Freeze everything
                keep_patterns=["head\\..*"],  # But keep head trainable
            )
        )

        # Only head should be trainable
        for name, param in self.model.named_parameters():
            if name.startswith("head"):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Should only return head params
        self.assertEqual(len(params), 2)  # head.weight and head.bias

    def test_partial_pattern_match(self):
        """Test that patterns use search (partial match)."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["weight"],  # Matches all weights
                keep_patterns=None,
            )
        )

        # Only biases should be trainable
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)

        # Returned params should only be biases
        bias_param_count = sum(
            1 for name, _ in self.model.named_parameters() if "weight" not in name
        )
        self.assertEqual(len(params), bias_param_count)


class TestFreezeWithNestedModel(unittest.TestCase):
    """Test freeze patterns with nested model structure."""

    def setUp(self):
        self.model = NestedModel()

    def test_freeze_vlm_module(self):
        """Test freezing entire VLM module."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\..*"],
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("vlm"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should only be action_model params
        action_model_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("action_model")
        )
        self.assertEqual(len(params), action_model_param_count)

    def test_freeze_specific_layers(self):
        """Test freezing specific layers by index."""
        # Freeze layers 0-2
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\.language\\.layers\\.[0-2]\\..*"],
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if (
                "vlm.language.layers.0" in name
                or "vlm.language.layers.1" in name
                or "vlm.language.layers.2" in name
            ):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Layers 3-4 should still be trainable
        for name, param in self.model.named_parameters():
            if "vlm.language.layers.3" in name or "vlm.language.layers.4" in name:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should exclude frozen layers
        trainable_param_count = sum(
            1 for name, param in self.model.named_parameters() if param.requires_grad
        )
        self.assertEqual(len(params), trainable_param_count)

    def test_freeze_vlm_keep_visual(self):
        """Test freezing VLM but keeping visual encoder trainable."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\..*"],
                keep_patterns=["vlm\\.visual\\..*"],
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("vlm.visual"):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            elif name.startswith("vlm"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Returned params should include visual and action_model params
        trainable_param_count = sum(
            1 for name, param in self.model.named_parameters() if param.requires_grad
        )
        self.assertEqual(len(params), trainable_param_count)


class TestApplyFreezeConfig(unittest.TestCase):
    """Test apply_freeze_config function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_none_config_returns_all_params(self):
        """With None config, should return all parameters."""
        params = apply_freeze_config(self.model, None)

        all_params = list(self.model.parameters())
        self.assertEqual(len(params), len(all_params))

    def test_with_freeze_config(self):
        """Test with a FreezeConfig-like object."""
        freeze_config = MagicMock()
        freeze_config.freeze_patterns = ["encoder\\..*"]
        freeze_config.keep_patterns = None

        params = apply_freeze_config(self.model, freeze_config)

        # Should only return non-encoder params
        encoder_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("encoder")
        )
        total_param_count = sum(1 for _ in self.model.parameters())

        self.assertEqual(len(params), total_param_count - encoder_param_count)


class TestLogTrainableParams(unittest.TestCase):
    """Test log_trainable_params function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_all_trainable(self):
        """Test logging when all params are trainable."""
        result = log_trainable_params(self.model)

        self.assertIn("trainable", result)
        self.assertIn("frozen", result)
        self.assertIn("encoder", result["trainable"])
        self.assertIn("decoder", result["trainable"])
        self.assertIn("head", result["trainable"])

    def test_partial_frozen(self):
        """Test logging with some frozen params."""
        # Freeze encoder
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False

        result = log_trainable_params(self.model)

        self.assertIn("encoder", result["frozen"])
        self.assertIn("decoder", result["trainable"])
        self.assertIn("head", result["trainable"])
        self.assertGreater(result["frozen"]["encoder"], 0)


class TestUnusedPatternWarnings(unittest.TestCase):
    """Test that unused patterns trigger warnings."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("flagscale.train.utils.optim_utils.logger")
    def test_warns_on_unused_freeze_pattern(self, mock_logger):
        """Should warn when freeze pattern matches nothing."""
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["nonexistent_module\\..*"],
                keep_patterns=None,
            )
        )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Freeze patterns matched NOTHING", warning_call)

    @patch("flagscale.train.utils.optim_utils.logger")
    def test_warns_on_unused_keep_pattern(self, mock_logger):
        """Should warn when keep pattern matches nothing."""
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=["nonexistent_module\\..*"],
            )
        )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Keep patterns matched NOTHING", warning_call)


class TestPrintParamNames(unittest.TestCase):
    """Test print_param_names debug helper."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("builtins.print")
    def test_prints_all_params(self, mock_print):
        """Should print all params when no pattern given."""
        print_param_names(self.model)

        self.assertGreater(mock_print.call_count, 0)

    @patch("builtins.print")
    def test_filters_by_pattern(self, mock_print):
        """Should only print params matching pattern."""
        print_param_names(self.model, pattern="encoder")

        # Should only print encoder params
        for call in mock_print.call_args_list:
            self.assertIn("encoder", call[0][0])


class TestParameterCounts(unittest.TestCase):
    """Test that parameter counts are correctly reported."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("flagscale.train.utils.optim_utils.logger")
    def test_parameter_count_logging(self, mock_logger):
        """Verify correct parameter counts are logged."""
        # Count total params
        total_params = sum(p.numel() for p in self.model.parameters())

        # Count encoder params
        encoder_params = sum(
            p.numel() for name, p in self.model.named_parameters() if name.startswith("encoder")
        )

        # Freeze encoder
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        # Check that info was logged with correct counts
        mock_logger.info.assert_called()
        info_call = mock_logger.info.call_args[0][0]
        self.assertIn(f"trainable={total_params - encoder_params:,}", info_call)
        self.assertIn(f"frozen={encoder_params:,}", info_call)
