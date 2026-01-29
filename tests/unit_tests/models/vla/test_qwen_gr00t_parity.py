import unittest

import torch
from omegaconf import OmegaConf

from flagscale.models.utils.constants import ACTION, OBS_STATE


class TestQwenGR00TParity(unittest.TestCase):
    """
    End-to-end parity test between QwenGR00T and QwenGR00T_V2.

    Note: This test requires GPU and the actual model weights.
    Skip in CI environments without GPU.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    def test_forward_parity(self):
        """Test that QwenGR00T_V2 produces same loss as QwenGR00T."""
        from flagscale.models.qwen_gr00t.qwen_gr00t import QwenGR00T
        from flagscale.models.vla.qwen_gr00t import QwenGR00T_V2

        # Create config
        config = self._create_test_config()

        # Create both models
        model_v1 = QwenGR00T(config=config).cuda()
        model_v2 = QwenGR00T_V2(config=config).cuda()

        # Copy action model weights from v1 to v2
        model_v2.action_model._head.load_state_dict(model_v1.action_model.state_dict())

        # Create test batch
        batch = self._create_test_batch()

        # Set same random seed for both
        torch.manual_seed(42)
        loss_v1 = model_v1.forward(batch)

        torch.manual_seed(42)
        loss_v2 = model_v2.forward(batch)

        # Compare losses
        self.assertTrue(
            torch.allclose(loss_v1, loss_v2, atol=1e-5),
            f"Loss mismatch: v1={loss_v1.item()}, v2={loss_v2.item()}",
        )

    def _create_test_config(self):
        """Create config matching examples/qwen_gr00t/conf/train/qwen_gr00t.yaml."""
        config_dict = {
            "model": {
                "model_name": "qwen_gr00t",
                "checkpoint_dir": "/workspace/models/Qwen/Qwen3-VL-4B-Instruct/",
                "vlm": {
                    "type": "qwen3-vl",
                },
                "qwenvl": {
                    "base_vlm": "/workspace/models/Qwen/Qwen3-VL-4B-Instruct/",
                    "attn_implementation": "flash_attention_2",
                    "vl_hidden_dim": 2048,
                },
                "action_model": {
                    "type": "flow_matching",
                    "action_model_type": "DiT-B",
                    "action_hidden_dim": 1024,
                    "hidden_size": 1024,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 8,
                    "future_action_window_size": 7,
                    "action_horizon": 8,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 4,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 2048,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
                "reduce_in_full_precision": True,
            },
            "data": {
                "data_path": "",
                "vla_data": {
                    "image_features": [
                        "observation.images.image",
                        "observation.images.wrist_image",
                    ],
                },
            },
            "system": {
                "batch_size": 16,
                "train_steps": 80000,
                "log_freq": 10,
                "grad_clip_norm": 1.0,
                "optimizer": {"name": "AdamW", "lr": 2.5e-5},
                "scheduler": {"warmup_steps": 5000},
                "checkpoint": {
                    "save_checkpoint": False,
                    "save_freq": 1000,
                    "output_directory": "/tmp",
                },
            },
        }
        return OmegaConf.create(config_dict)

    def _create_test_batch(self):
        """
        Create test batch matching actual training data format.

        Actual batch structure:
        - action: [16, 8, 7] float32
        - task: list of 16 strings
        - observation.images.wrist_image: [16, 3, 224, 224] float32
        - observation.images.image: [16, 3, 224, 224] float32
        - observation.state: [16, 1, 8] float32
        """
        batch_size = 16
        action_horizon = 8
        action_dim = 7
        state_dim = 8
        img_channels = 3
        img_size = 224

        return {
            ACTION: torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32),
            "task": ["put the bowl on the plate"] * batch_size,
            "observation.images.image": torch.randn(
                batch_size, img_channels, img_size, img_size, dtype=torch.float32
            ),
            "observation.images.wrist_image": torch.randn(
                batch_size, img_channels, img_size, img_size, dtype=torch.float32
            ),
            OBS_STATE: torch.randn(batch_size, 1, state_dim, dtype=torch.float32),
        }


if __name__ == "__main__":
    unittest.main()
