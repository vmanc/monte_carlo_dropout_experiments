from dataclasses import dataclass, field
from typing import List
import numpy.typing as npt
import numpy as np
import torch

@dataclass
class ExperimentalConfig:
    # Dataset Configuration
    dataset_name: str = "cardiffnlp/x_sensitive"
    text_col: str = "text"
    label_cols: List[str] = field(default_factory=lambda: [
        "conflictual", "profanity", "sex", "drugs", "selfharm", "spam"
    ])
    num_labels: int = field(init=False)
    max_seq_length: int = 96
     
    model_name: str = "roberta-base"

    # Experimental grid parameters
    dropout_grid: npt.NDArray[np.float16] = field(
        default_factory=lambda: np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    )
    t_grid: npt.NDArray[np.int16] = field(
        default_factory=lambda: np.array([5, 10, 20, 50, 100], dtype=np.int16)
    )

    # Training Arguments
    output_dir: str = f"./experimental_results/{dataset_name}/{model_name}"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_steps: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.0
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    bf16: bool = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    fp16: bool = not bf16
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    evaluation_strategy: str = "epoch"
    # Callback Arguments
    early_stopping_patience: int = 3 
    # Seed for reproducibility
    random_seed: int = 42

    def __post_init__(self):
        self.num_labels = len(self.label_cols)