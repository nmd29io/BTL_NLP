# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Cấu hình mô hình"""
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 5000
    pad_idx: int = 0

@dataclass
class TrainingConfig:
    """Cấu hình training"""
    # Data
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    batch_size: int = 32
    max_len: int = 64
    min_freq: int = 2
    
    # Optimizer
    learning_rate: float = 0.0001
    warmup_steps: int = 4000
    clip_grad: float = 1.0
    weight_decay: float = 0.0001
    
    # Training
    max_epochs: int = 100
    max_time_hours: float = 1.0
    patience: int = 10
    save_every: int = 5
    
    # Evaluation
    beam_size: int = 5
    
    # System
    use_amp: bool = True
    num_workers: int = 4

# Các cấu hình mẫu
SMALL_CONFIG = TrainingConfig(
    batch_size=32,
    d_model=256,
    n_heads=4,
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=1024,
    learning_rate=0.0001
)

MEDIUM_CONFIG = TrainingConfig(
    batch_size=64,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    learning_rate=0.0001
)

# Export
__all__ = ['ModelConfig', 'TrainingConfig', 'SMALL_CONFIG', 'MEDIUM_CONFIG']