import os
import json
import torch

from torch.utils.data import Dataset

class PAN23Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(path))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        file_path = os.path.join(self.path, f"{index}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

class PAN23CollatorFn:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [item[label] for item in batch for label in ["text1", "text2"]]  # (2*batch_size,)
        labels = [item["label"] for item in batch]  # (batch_size,)

        encoding = self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=torch.tensor(labels),
        )