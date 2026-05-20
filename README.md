# Monte Carlo Dropout Experiments

Uncertainty quantification on multi-label sensitive content classification using Monte Carlo Dropout with RoBERTa-base on the CardiffNLP X-Sensitive dataset.

## Overview

This repository implements Monte Carlo (MC) Dropout for epistemic uncertainty estimation in transformer-based multi-label text classification. The model is fine-tuned with dropout enabled, and at inference time dropout is kept active across T stochastic forward passes. The predictive mean and predictive variance across these passes are used as the prediction and its uncertainty.

The experiments sweep two hyperparameters: the dropout rate p applied to both hidden and attention layers during training, and the number of forward passes T at inference. Results are saved per (p, T) configuration for downstream analysis.

## Method

For each (p, T) cell in the experimental grid:

1. Fine-tune RoBERTa-base on the training set with `hidden_dropout_prob = attention_probs_dropout_prob = p` and binary cross-entropy loss applied independently per label (the standard for multi-label classification).
2. At inference, enable dropout in all `nn.Dropout` modules and run T forward passes over the test set.
3. Apply sigmoid to each pass to obtain per-label probabilities, then compute the predictive mean and predictive variance per (instance, label).
4. Save the full probability matrix, predictive mean, and predictive variance to HDF5.

The variance per (instance, label) is computed as the sample second moment minus the sample mean squared across the T passes: `(1/T) * sum(p_t^2) - ((1/T) * sum(p_t))^2`, treating each label independently.

## Dataset

CardiffNLP X-Sensitive, a multi-label dataset with six categories: conflictual, profanity, sex, drugs, selfharm, spam. The dataset is loaded directly from Hugging Face.

## Experimental Grid

- Dropout rates: 0.1, 0.2, 0.3, 0.4, 0.5
- Forward passes T: 5, 10, 20, 50, 100

This yields 25 configurations. The script is designed to run one configuration per task, indexed by a SLURM task ID, so the full grid can be parallelized across cluster jobs.

## Installation

```bash
git clone <repo-url>
cd monte_carlo_dropout_experiments
pip install -r requirements.txt
```

A GPU is recommended. The code will use bf16 if supported, otherwise fp16 if CUDA is available, otherwise fp32.

### Dataset access

`cardiffnlp/x_sensitive` is a gated dataset on Hugging Face. You will need to request access on the dataset page and authenticate before running. Set `HF_TOKEN` in your environment (or run `huggingface-cli login`) so that `load_dataset` can pull the data.

## Usage

Run a single grid cell by SLURM task ID (0 to 24):

```bash
cd src
python main.py --task-id 0
```

The task ID maps to (dropout rate, T) as `dropout_grid[task_id // len(t_grid)], t_grid[task_id % len(t_grid)]`.

Run standard inference without MC Dropout for a given training dropout rate (baseline comparison):

```bash
cd src
python main.py --no-dropout-inference 0.0
```

This trains a fresh model at the specified dropout rate and runs a single deterministic forward pass at inference. It does not reload weights from a prior MC Dropout run, so the comparison isolates the inference procedure given an independently trained model rather than holding the trained weights fixed.

## Output

Results are written to `experimental_results/<dataset_name>/<model_name>/dropout_<rate>/T_<T>/mc_dropout_results.h5` with the following datasets and attributes:

- `predictive_mean` (N x C): mean predicted probability per instance and label
- `predictive_variance` (N x C): variance across T forward passes
- `probas_matrix` (T x N x C): the raw probabilities from each forward pass
- Attributes: `dropout_rate`, `T`, `label_cols`

## Project Structure

```
src/
  main.py                          Entry point with CLI
  experimental_config.py           Dataclass holding all experiment hyperparameters
  experiment_runner.py             Training and MC Dropout inference logic
  data_loading/
    data_loader.py                 Dataset loading, tokenization, label assembly
  model_utils/
    model_utils.py                 Model construction and dropout context manager
```

## Notes

- `max_length` for tokenization is set to the 95th percentile of token counts across train and test, rather than a fixed value. This limits truncation on long examples while avoiding excessive padding on short ones. The `max_seq_length` field on `ExperimentalConfig` is not read by the data loader and exists only as a historical default; the tokenizer length comes from the 95th percentile pass. Will be fixed in the next iteration.
- The `warmup_steps` field on `ExperimentalConfig` is passed to the Hugging Face `TrainingArguments` `warmup_ratio` argument. The value (0.1) is interpreted as a fraction of total training steps, not a step count. The field name is retained for backward compatibility with earlier configs. Will be fixed in the next iteration.
- The `enable_dropout` context manager sets only `nn.Dropout` modules to train mode during inference, leaving the rest of the model in eval mode. This is the correct behavior for MC Dropout and avoids changing batch norm or other train-mode-only layers.
- `MCDropoutTrainer.prediction_step` applies `enable_dropout` to every prediction call, which includes the per-epoch validation pass during training. This is intentional: model selection and early stopping operate under the same stochastic inference procedure used at test time, so the checkpoint chosen by `load_best_model_at_end` is the one that performs best under MC Dropout inference. Note that the per-epoch validation loss is a single stochastic sample rather than a T-pass average, so the early-stopping signal is noisier than a deterministic validation loss would be, and the noise grows with the dropout rate.
- Early stopping is used during training with patience 3 on validation loss.
