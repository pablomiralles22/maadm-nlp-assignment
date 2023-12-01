import os
import json
import torch
import pytorch_lightning as pl

from typing import Union
from torch.utils.data import Dataset, DataLoader
from utils.custom_types import TokenizerEncoding
from transformers import AutoTokenizer

# =================================================================================================== #
# =========================================== PAN23Dataset ========================================== #
# =================================================================================================== #

class PAN23Dataset(Dataset):
    """
    A custom dataset class for loading PAN23 data.

    Args:
        path (str): The path to the directory containing the data files.

    Attributes:
        path (str): The path to the directory containing the data files.
        len (int): The number of data files in the directory.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data item at the specified index.
        get_positive_ratio(): Returns the ratio of positive labels in the dataset.
    """

    def __init__(self, path):
        """
        Initializes a new instance of the PAN23Dataset class.

        Args:
            path (str): The path to the directory containing the data files.
        """
        self.path = path
        self.len = len(os.listdir(path))

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of data files in the directory.
        """
        return self.len

    def __getitem__(self, index):
        """
        Returns the data item at the specified index.

        Args:
            index (int): The index of the data item to retrieve.

        Returns:
            dict: The data item at the specified index.
        """
        file_path = os.path.join(self.path, f"{index}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_positive_ratio(self):
        """
        Returns the ratio of positive labels in the dataset. This is
        useful for correcting the class imbalance in the dataset during
        training.

        Returns:
            float: The ratio of positive labels in the dataset.
        """
        cnt = 0
        for i in range(len(self)):
            item = self[i]
            if item["label"] == 1:
                cnt += 1
        return cnt / len(self)

# =================================================================================================== #
# =========================================== PAN23CollatorFn ======================================= #
# =================================================================================================== #

class PAN23CollatorFn:
    """
    A collator function for processing data batches in the PAN23 dataset.

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode the texts.
        max_len (int): The maximum length of the encoded texts.

    Returns:
        dict: A dictionary containing the encoded texts and labels.

    """

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        """
        Process a batch of data.

        Args:
            batch (list): A list of data items.

        Returns:
            dict: A dictionary containing the tokenized texts and labels. We return
                both disjoint and joint encodings of the texts, that is, each text pair
                is tokenized as a pair and as two different texts. Then, each model
                will use the encoding that is most appropriate for it.
        """
        disjoint_texts = [item[label] for item in batch for label in ["text1", "text2"]]  # (2*batch_size,)
        joint_texts = [(item["text1"], item["text2"]) for item in batch]  # (batch_size,)
        labels = [item["label"] for item in batch]  # (batch_size,)

        disjoint_encoding_dict = self.__encode(disjoint_texts)
        joint_encoding_dict = self.__encode(joint_texts)

        return dict(
            disjoint_encoding=TokenizerEncoding(
                input_ids=disjoint_encoding_dict["input_ids"],
                attention_mask=disjoint_encoding_dict["attention_mask"],
                token_type_ids=disjoint_encoding_dict["token_type_ids"],
            ),
            joint_encoding = TokenizerEncoding(
                input_ids=joint_encoding_dict["input_ids"],
                attention_mask=joint_encoding_dict["attention_mask"],
                token_type_ids=joint_encoding_dict["token_type_ids"],
            ),
            labels=torch.tensor(labels),
        )

    def __encode(self, texts: list[Union[str, tuple[str, str]]]):
        """
        Encode the texts using the tokenizer.

        Args:
            texts (list): A list of texts or pairs of texts to be encoded.

        Returns:
            dict: A dictionary containing the encoded texts.

        """
        return self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

# =================================================================================================== #
# =========================================== PAN23DataModule ======================================= #
# =================================================================================================== #

class PAN23DataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading PAN23 dataset.

    Args:
        data_path (str): The path to the PAN23 dataset.
        collator_config (dict): Configuration for the collator function.
        loader_config (dict): Configuration for the data loader.

    Attributes:
        data_path (str): The path to the PAN23 dataset.
        tokenizer (AutoTokenizer): The tokenizer object used for tokenization.
        loader_config (dict): Configuration for the data loader.
        train_dataset (PAN23Dataset): The training dataset.
        val_dataset (PAN23Dataset): The validation dataset.
    """

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
        """
        Create an instance of PAN23DataModule from a single dictionary of configuration.
        This dictionary is split into two parts: one for the collator function and one
        for the data loader.

        Args:
            config (dict): The joint configuration.

        Returns:
            PAN23DataModule: An instance of PAN23DataModule.
        """
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name,
        )
        collator_config["tokenizer"] = self.tokenizer

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
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: A DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: A DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            shuffle=False,
        )

    def get_positive_ratio(self):
        """
        Returns the ratio of positive examples of the training dataset.

        Returns:
            float: The positive ratio of the training dataset.
        """
        return self.train_dataset.get_positive_ratio()

    def get_padding_idx(self):
        """
        Returns the padding index of the tokenizer.

        Returns:
            int: The padding index of the tokenizer.
        """
        return self.tokenizer.pad_token_id

    def get_vocab_size(self):
        """
        Returns the vocabulary size of the tokenizer.

        Returns:
            int: The vocabulary size of the tokenizer.
        """
        return self.tokenizer.vocab_size