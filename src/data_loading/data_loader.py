import os
from typing import Tuple, List, Union, Dict, Any
import numpy as np
from transformers import AutoTokenizer
import torch
from scipy.sparse import csr_matrix
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset, IterableDatasetDict

class DataLoader:
    def __init__(self, 
                dataset_name: str,
                text_col: str,
                label_cols: List[str], 
                tokenizer_name: str = "bert-base-uncased",
                percentile: float = 95.0):
        
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.label_cols = label_cols
        self.tokenizer_name = tokenizer_name
        self.percentile: float = percentile
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_length: int = -1
        self.num_classes: int = len(self.label_cols)
    
    def _compute_percentile(self, all_texts: List[str]) -> int:
        lengths: List[int] = []
        for text in all_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            lengths.append(len(tokens))
        
        return int(np.percentile(lengths, self.percentile))
    
    def _tokenize_batch(self, 
                        examples: dict) -> AutoTokenizer:
        tokenized_batch = self.tokenizer(
            examples[f"{self.text_col}"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        #for col in self.label_cols:
        #    tokenized_batch[col] = examples[col]

        # Transpose the matrix so that it becomes len(examples) x len(label_cols)  
        label_matrix = np.stack([examples[col] for col in self.label_cols]).T 
        tokenized_batch["labels"] = label_matrix.astype(np.float32).tolist()
        return tokenized_batch   
    
    def load(self) -> Tuple[
            Dataset,        
            csr_matrix,    
            Dataset,       
            csr_matrix,     
            int             
        ]:  
            
            if self.dataset_name == "google/jigsaw_toxicity_pred":
                script_dir = os.path.dirname(os.path.abspath(__file__))

                dataset = load_dataset(f"{script_dir}/jigsaw_toxicity_pred.py", data_dir=f"{script_dir}/{self.dataset_name}", trust_remote_code=True)
                ds_train: Dataset = dataset['train'] # type: ignore
                ds_test: Dataset = dataset['test']  # type: ignore
            else:
                ds_train: Dataset = load_dataset(self.dataset_name, split="train") # type: ignore
                ds_test: Dataset = load_dataset(self.dataset_name, split="test") # type: ignore
            raw_train_texts: List[str] = list(ds_train[f"{self.text_col}"]) # type: ignore
            raw_test_texts: List[str] = list(ds_test[f"{self.text_col}"])  # type: ignore
            all_texts: List[str] = raw_train_texts + raw_test_texts



            one_hot_train_labels: csr_matrix = csr_matrix(
                np.vstack(
                    [np.array(ds_train[col], dtype=int)  # type: ignore
                    for col in self.label_cols]
                ).T
            ) # type: ignore
            one_hot_test_labels: csr_matrix = csr_matrix(
                np.vstack(
                    [np.array(ds_test[col], dtype=int)  # type: ignore
                    for col in self.label_cols]
                ).T
            ) # type: ignore
            

            self.max_length = self._compute_percentile(all_texts)
            train_dataset_tokenized: Dataset = ds_train.map(
                self._tokenize_batch,
                batched=True,
                remove_columns=ds_train.column_names
            ) #type: ignore

            test_dataset_tokenized: Dataset = ds_test.map(
                self._tokenize_batch,
                batched=True,
                remove_columns=ds_test.column_names
            ) #type: ignore
            train_dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            test_dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            return (
                train_dataset_tokenized,
                one_hot_train_labels,
                test_dataset_tokenized,
                one_hot_test_labels,
                self.num_classes,
            )