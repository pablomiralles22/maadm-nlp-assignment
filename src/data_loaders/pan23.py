import os
import json
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

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

    def get_positive_ratio(self):
        cnt = 0
        for i in range(len(self)):
            item = self[i]
            if item["label"] == 1:
                cnt += 1
        return cnt / len(self)


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


class PAN23DataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "tokenizer": "roberta-base",
            "max_len": 512,
        }

    @classmethod
    def get_default_loader_config(cls):
        return {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
        }

    @classmethod
    def from_joint_config(cls, config):
        data_path = config.pop("data_path")
        collator_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_collator_config()
        }
        loader_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_loader_config()
        }
        return cls(data_path, collator_config, loader_config)

    def __init__(self, data_path: str, collator_config: dict, loader_config: dict):
        super().__init__()
        self.data_path = data_path

        # build collator_fn
        collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }
        ## change tokenizer name to tokenizer object
        pretrained_tokenizer_name = collator_config.pop("tokenizer")
        collator_config["tokenizer"] = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name,
        )

        # build loader config
        self.loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": PAN23CollatorFn(**collator_config),
        }

        # load datasets
        self.train_dataset = PAN23Dataset(os.path.join(data_path, "train"))
        self.val_dataset = PAN23Dataset(os.path.join(data_path, "validation"))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            shuffle=False,
        )

    def get_positive_ratio(self):
        return self.train_dataset.get_positive_ratio()
