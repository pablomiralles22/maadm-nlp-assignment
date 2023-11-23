import os
import random
import re
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

class BlogDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(path))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        author_path = os.path.join(self.path, str(index))
        author_files = os.listdir(author_path)

        post1_ind, post2_ind = random.sample(author_files, 2)
        post1_path = os.path.join(author_path, post1_ind)
        post2_path = os.path.join(author_path, post2_ind)

        with open(post1_path, "r", encoding="utf-8") as f:
            post1 = f.read()
        with open(post2_path, "r", encoding="utf-8") as f:
            post2 = f.read()

        return {
            "post1": post1,
            "post2": post2,
        }

class BlogCollatorFn:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        posts = []
        for item in batch:
            posts.append(self.__clean_text(item["post1"]))
            posts.append(self.__clean_text(item["post2"]))

        encoding = self.tokenizer.batch_encode_plus(
            posts,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    def __clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        if len(words) > self.max_len:
            start = random.randint(0, len(words) - self.max_len)
            words = words[start:start + self.max_len]
        return " ".join(words)

class BlogDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "pretrained_tokenizer_name": "roberta-base",
            "max_len": 512,
        }

    @classmethod
    def get_default_loader_config(cls):
        return {
            "batch_size": 64,
            "num_workers": 4,
            "pin_memory": True,
        }

    @classmethod
    def from_joint_config(cls, config):
        data_path = config.pop("data_path")
        val_ratio = config.pop("val_ratio")
        collator_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_collator_config()
        }
        loader_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_loader_config()
        }
        return cls(data_path, val_ratio, collator_config, loader_config)

    def __init__(self, data_path: str, val_ratio: float, collator_config: dict, loader_config: dict):
        super().__init__()
        self.data_path = data_path

        # build collator_fn
        collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }
        ## change tokenizer name to tokenizer object
        pretrained_tokenizer_name = collator_config.pop("pretrained_tokenizer_name")
        collator_config["tokenizer"] = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name,
        )

        # build loader config
        self.loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": BlogCollatorFn(**collator_config),
        }

        # load datasets
        dataset = BlogDataset(os.path.join(data_path))
        self.train_dataset, self.val_dataset = self.__split_torch_dataset(dataset, val_ratio)

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

    def __split_torch_dataset(self, dataset, val_ratio):
        val_size = int(val_ratio * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset,
            [len(dataset) - val_size, val_size],
        )
        return train_dataset, val_dataset