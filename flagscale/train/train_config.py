"""
Training configuration models using Pydantic.
"""

from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    name: str = "AdamW"
    lr: float = 2.5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""

    warmup_steps: int = 1000
    decay_steps: int = 30000
    decay_lr: float = 2.5e-6


class CheckpointConfig(BaseModel):
    """Checkpoint saving configuration"""

    save_checkpoint: bool = True
    save_freq: int = 1000
    output_directory: str


class SystemConfig(BaseModel):
    """Training loop configuration"""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    batch_size: int = 1
    train_steps: int = 100000
    log_freq: int = 10
    grad_clip_norm: float = 1.0
    use_amp: bool = False
    shuffle: bool = False
    num_workers: int = 4

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)


class DataConfig(BaseModel):
    """Dataset configuration"""

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    data_path: str = Field(..., description="Path to training dataset")
    tolerance_s: float = 0.0001
    use_imagenet_stats: bool = True
    rename_map: dict[str, str] | None = None
    use_quantiles: bool = False
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)


class ModelConfig(BaseModel):
    """Model configuration.

    This accepts any model-specific fields dynamically, allowing any other model config directly from YAML.

    Required fields:
    - model_name: Which model to use ('pi0' or 'pi0.5')
    - checkpoint_dir: Path to pretrained checkpoint (for loading weights)

    All other fields are passed through to the model's config class.
    """

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }  # Allow extra fields for model-specific config

    # Required fields to identify which model and checkpoint to use
    model_name: str = Field(..., description="Model name: 'pi0' or 'pi0.5'")
    checkpoint_dir: str = Field(..., description="Path to pretrained model checkpoint")
    raw: DictConfig | None = Field(default=None, exclude=True)

    def __getattr__(self, name):
        raw = self.__dict__.get("raw")
        if raw is not None and hasattr(raw, name):
            return getattr(raw, name)
        raise AttributeError(name)

    # @field_validator("model_name")
    # @classmethod
    # def validate_model_name(cls, v):
    #     if v not in ["pi0", "pi0.5"]:
    #         raise ValueError(f"Invalid model_name: {v}. Must be 'pi0' or 'pi0.5'")
    #     return v

    def get_model_config_dict(self) -> dict[str, Any]:
        """Get all model-specific config fields (excluding train-level fields)."""
        return self.model_dump(exclude={"model_name", "checkpoint_dir"})


class TrainConfig(BaseModel):
    """Top-level training configuration for native backend"""

    system: SystemConfig
    model: ModelConfig
    data: DataConfig

    @classmethod
    def from_hydra_config(cls, hydra_config) -> "TrainConfig":
        """Convert Hydra DictConfig to Pydantic TrainConfig"""
        train = hydra_config.train
        train_dict = OmegaConf.to_container(train, resolve=True)
        train_dict["system"] = SystemConfig(**train_dict["system"], raw=train.system)
        train_dict["data"] = DataConfig(**train_dict["data"], raw=train.data)
        train_dict["model"] = ModelConfig(**train_dict["model"], raw=train.model)
        return cls(**train_dict)

    class Config:
        # Allow arbitrary types for complex objects
        arbitrary_types_allowed = True
