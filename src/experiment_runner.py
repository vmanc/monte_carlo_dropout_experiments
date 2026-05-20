import numpy as np
import numpy.typing as npt
import torch
import os
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.trainer import (
    Trainer,
)
from datasets import Dataset, DatasetDict
from transformers.training_args import (
    TrainingArguments
)
from experimental_config import ExperimentalConfig
from data_loading.data_loader import DataLoader
from model_utils.model_utils import get_model, enable_dropout
from transformers.trainer_callback import EarlyStoppingCallback
import h5py
class MCDropoutTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with enable_dropout(model):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

class ExperimentRunner:
    def __init__(self, experimental_config: ExperimentalConfig) -> None:
        self.train_dataset: Dataset 
        self.test_dataset: Dataset
        self.experimental_config : ExperimentalConfig = experimental_config
        self.num_classes: int = -1
        self.is_in_main_proc: bool = False
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self.is_in_main_proc = True
    
    def load_data(self):
        if self.is_in_main_proc:
            print("Pre-processing Data: Tokenizing the dataset")
        
        data_loader = DataLoader(
            dataset_name=self.experimental_config.dataset_name,
            text_col=self.experimental_config.text_col,
            label_cols=self.experimental_config.label_cols,
            tokenizer_name=self.experimental_config.model_name
        )

        self.train_dataset, _, self.test_dataset, _, self.num_classes = data_loader.load()

    def retrieve_training_configuration(self, dropout_rate: float) -> TrainingArguments:

        output_dir = os.path.join(
            self.experimental_config.output_dir,
            f"dropout_{dropout_rate:.2f}"
        )

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.experimental_config.learning_rate,
            weight_decay=self.experimental_config.weight_decay,
            lr_scheduler_type=self.experimental_config.lr_scheduler_type,
            warmup_ratio=self.experimental_config.warmup_steps,
            max_grad_norm=self.experimental_config.max_grad_norm,
            label_smoothing_factor=self.experimental_config.label_smoothing_factor,
            num_train_epochs=self.experimental_config.num_train_epochs,
            per_device_train_batch_size=self.experimental_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.experimental_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.experimental_config.gradient_accumulation_steps,
            bf16=self.experimental_config.bf16,
            fp16=self.experimental_config.fp16,
            load_best_model_at_end=self.experimental_config.load_best_model_at_end,
            metric_for_best_model=self.experimental_config.metric_for_best_model,
            eval_strategy=self.experimental_config.evaluation_strategy,
            save_strategy=self.experimental_config.evaluation_strategy,
            seed=self.experimental_config.random_seed,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=50,
        )
    
    def create_validation_set(self, validation_size = 0.1) -> Tuple[Dataset, Dataset]:
        train_val_split = self.train_dataset.train_test_split(test_size=validation_size, seed=self.experimental_config.random_seed)
        return train_val_split["train"], train_val_split["test"]
    
    def train_model(self, dropout_rate: float) -> MCDropoutTrainer:
        if self.is_in_main_proc:
            print(f"[INFO] IN TRAINING DROPOUT RATE: {dropout_rate}")
        
        sequence_classification_model = get_model(
            model_name=self.experimental_config.model_name,
            num_classes=self.num_classes,
            dropout_probability=dropout_rate
        )

        training_arguments = self.retrieve_training_configuration(dropout_rate)
        training_set, validation_set = self.create_validation_set()
        sequence_classification_trainer = MCDropoutTrainer(
            model=sequence_classification_model,
            args=training_arguments,
            train_dataset=training_set,
            eval_dataset=validation_set,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.experimental_config.early_stopping_patience
                )
            ]
        )
        sequence_classification_trainer.train()
        return sequence_classification_trainer
    
    def compute_predictive_mean_and_variance(self, sequential_classification_trainer: MCDropoutTrainer, T: int) -> Dict[str, Optional[npt.NDArray[np.float32]]]:
        if self.is_in_main_proc:
            print(f"Running MC Dropout inference with T={T} forward passes")
        
        probas_list: List[npt.NDArray[np.float32]] = []

        for num_forward_passes in tqdm(range(T), desc=f"MC Dropout (T={T})", disable=not self.is_in_main_proc):
            predictions = sequential_classification_trainer.predict(self.test_dataset)  # type: ignore[arg-type]
            probas = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
            probas_list.append(probas)
        
        if len(probas_list) == 0:
            print("[ERROR] Cannot calculate predictive mean and variance. probas are empty")
            return {
                "predictive_mean": None,
                "predictive_variance": None,
                "probas_matrix": None,
            }

        probas_matrix = np.stack(probas_list, axis=0)
        predictive_mean = np.mean(probas_matrix, axis=0)
        predictive_variance = np.mean(probas_matrix ** 2, axis=0) - predictive_mean ** 2
        return {
            "predictive_mean": predictive_mean,
            "predictive_variance": predictive_variance,
            "probas_matrix": probas_matrix,
        }
    
    def run_standard_inference(self, dropout_rate: float = 0.0) -> None:
        if self.is_in_main_proc:
            print(f"Training and running inference with dropout={dropout_rate:.2f} (no dropout)")

        trainer = self.train_model(dropout_rate)

        predictions = trainer.predict(self.test_dataset)  # type: ignore[arg-type]
        probas = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()

        if self.is_in_main_proc:
            save_dir = os.path.join(
                self.experimental_config.output_dir,
                f"dropout_{dropout_rate:.2f}",
                "standard_inference"
            )
            os.makedirs(save_dir, exist_ok=True)

            with h5py.File(os.path.join(save_dir, "standard_inference.h5"), "w") as f:
                f.attrs["dropout_rate"] = dropout_rate
                f.attrs["label_cols"] = self.experimental_config.label_cols
                f.create_dataset("probas", data=probas, compression="gzip")

            print(f"Saved to {save_dir}/standard_inference.h5")

        del trainer
        torch.cuda.empty_cache()
        
    def run_experiment(self, slurm_task_id: int) -> Dict[str, Optional[npt.NDArray[np.float32]]]:
        forward_passes_grid_len = self.experimental_config.t_grid.size
        selected_dropout_rate = float(self.experimental_config.dropout_grid[slurm_task_id // forward_passes_grid_len])
        selected_num_forward_passes = int(self.experimental_config.t_grid[slurm_task_id % forward_passes_grid_len])
        if self.is_in_main_proc:
            print(f"Task {slurm_task_id}: dropout={selected_dropout_rate:.2f}, T={selected_num_forward_passes}")
        
        sequential_classification_trainer = self.train_model(selected_dropout_rate)
        mc_dropout_results = self.compute_predictive_mean_and_variance(
            sequential_classification_trainer, 
            selected_num_forward_passes
        )
        if self.is_in_main_proc:
            save_dir = os.path.join(
                self.experimental_config.output_dir,
                f"dropout_{selected_dropout_rate:.2f}",
                f"T_{selected_num_forward_passes}"
            )
            os.makedirs(save_dir, exist_ok=True)
            
            with h5py.File(os.path.join(save_dir, "mc_dropout_results.h5"), "w") as results_file:
                results_file.attrs["dropout_rate"] =  selected_dropout_rate
                results_file.attrs["T"] =  selected_num_forward_passes
                results_file.attrs["label_cols"] = self.experimental_config.label_cols
                for k, v in mc_dropout_results.items():
                    if v is not None:
                        results_file.create_dataset(k, data=v, compression="gzip")
        
        del sequential_classification_trainer
        torch.cuda.empty_cache()

        return mc_dropout_results







