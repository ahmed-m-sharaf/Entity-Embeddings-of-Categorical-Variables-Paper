"""Configuration classes for baseline and ablation experiments."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BaselineConfig:
    """Original Model Random Based configuration."""
    name: str = "baseline"
    
    # Data
    split_type: str = "random"
    shuffle_loader: bool = True
    test_size: float = 0.1
    
    # Training
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    criterion_type: str = "mse"
    
    # Model architecture
    output_sigmoid: bool = True
    batch_norm: bool = False
    hidden_sizes: tuple = (1000, 500)
    
    # Default Embedding dimensions
    emp_dims: Dict[str, int] = None
    
    # Misc
    seed: int = 42
    device: str = "cuda"
    ensemble_size: int = 5

    def __post_init__(self):
        if self.emp_dims is None:
            self.emp_dims = {
                "Store": 10,
                "DayOfWeek": 6,
                "day": 10,
                "month": 6,
                "year": 2,
                "Promo": 1,
                "State": 6,
            }


@dataclass
class AblationTimeBasedConfig(BaselineConfig):
    """Original Model Time Based."""
    name: str = "ablation_time_based"
    split_type: str = "time"
    shuffle_loader: bool = False


@dataclass
class AblationEmbeddingConfig(BaselineConfig):
    """Ablation With Embedding Dim."""
    name: str = "ablation_embedding"
    
    def __post_init__(self):
        self.emp_dims = {
            "Store": 20,
            "DayOfWeek": 12,
            "day": 20,
            "month": 12,
            "year": 4,
            "Promo": 2,
            "State": 12,
        }


@dataclass
class AblationBatchNormConfig(BaselineConfig):
    """Ablation With Batch Normalization."""
    name: str = "ablation_bn"
    batch_norm: bool = True


@dataclass
class AblationDecLRConfig(BaselineConfig):
    """Ablation With Learning Rate Decreasing."""
    name: str = "ablation_dec_lr"
    learning_rate: float = 0.0005


@dataclass
class AblationIncLRConfig(BaselineConfig):
    """Ablation With Learning Rate Increasing."""
    name: str = "ablation_inc_lr"
    learning_rate: float = 0.005


def get_config(config_name: str) -> BaselineConfig:
    """Get configuration by name."""
    configs = {
        'baseline': BaselineConfig(),
        'ablation_time_based': AblationTimeBasedConfig(),
        'ablation_embedding': AblationEmbeddingConfig(),
        'ablation_bn': AblationBatchNormConfig(),
        'ablation_dec_lr': AblationDecLRConfig(),
        'ablation_inc_lr': AblationIncLRConfig(),
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    return configs[config_name]
