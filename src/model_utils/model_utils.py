from transformers import AutoModelForSequenceClassification, AutoConfig, RobertaForSequenceClassification
import torch.nn as nn
import numpy as np
from contextlib import contextmanager


def get_model(model_name: str, num_classes: int, dropout_probability: float):
    model_configuration = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        hidden_dropout_prob=dropout_probability,
        attention_probs_dropout_prob=dropout_probability,
        problem_type="multi_label_classification" 
    )
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=model_configuration)

@contextmanager
def enable_dropout(sequence_classification_model: nn.Module):
    modified_layers = []

    for layer in sequence_classification_model.modules():
        if isinstance(layer, nn.Dropout) and not layer.training:
            layer.train()
            modified_layers.append(layer)
    try:
        yield 
    except Exception as e:
        print(f"Unable to return back to the original context. Error: {e}")
        raise
    finally:
        for layer in modified_layers:
            layer.eval()