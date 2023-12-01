import os
import sys
import torch
import json
import argparse
import numpy as np

from torch.utils.data import Dataset

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.model_builder import ModelBuilder
from models.base_model import BaseModel
from heads.classification_head import ModelWithClassificationHead
from data_loaders.pan23 import PAN23DataModule, PAN23CollatorFn
from utils.merge_dicts import merge_dicts


class OriginalPAN23Dataset(Dataset):
    """
    A custom dataset class for loading PAN23 data, as given in its original format.
    This is useful for evaluating the performance of a model with the same setting
    as in other works. Each index in this dataset represents a single document.

    Args:
        path (str): The path to the directory containing the data files.

    Attributes:
        path (str): The path to the directory containing the data files.
        len (int): The number of data files in the directory.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data item at the specified index.
    """

    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(path)) // 2  # two files per doc

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        labels = self.__get_truths(index)
        paragraphs = self.__get_paragraphs(index)

        return [
            { "label": label, "text1": text1, "text2": text2 }
            for label, text1, text2 in zip(labels, paragraphs, paragraphs[1:])
        ]

    def __get_truths(self, index: int) -> list[int]:
        path = os.path.join(self.path, f"truth-problem-{index+1}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["changes"]

    def __get_paragraphs(self, index: int) -> list[str]:
        path = os.path.join(self.path, f"problem-{index+1}.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()

def build_validation_dataset_for_task(task: int, path: str) -> OriginalPAN23Dataset:
    path = os.path.join(
        path,
        f"pan23-multi-author-analysis-dataset{task}/pan23-multi-author-analysis-dataset{task}-validation",
    )
    return OriginalPAN23Dataset(path)

def predict(model: BaseModel, batch: dict):
    labels = batch["labels"].float().view(-1, 1)
    logits = model.forward(batch["joint_encoding"], batch["disjoint_encoding"])
    predictions = torch.sigmoid(logits) > 0.5
    return predictions, labels


@torch.no_grad()
def evaluate(model: BaseModel, dataset: OriginalPAN23Dataset, collator_fn: PAN23CollatorFn):
    f1_scores = []
    model.eval()
    for idx in range(len(dataset)):
        batch = dataset[idx]
        collated_batch = collator_fn(batch)
        predictions, labels = predict(model, collated_batch)
        f1_score = (2 * (predictions * labels).sum() / (predictions.sum() + labels.sum())).item()
        f1_scores.append(f1_score)
        if idx % 50 != 0:
            continue
        print(f"Average F1 score up to document {idx+1}: {np.mean(f1_scores):.4f}")
    print(f"Average F1 score: {np.mean(f1_scores):.4f)}")

def load_model_and_data_module(
    model_config: dict, checkpoint_path: str, task: str
):
    """
    Takes the checkpoint of a pytorch lightning module. It must be a model
    with a classification head. Loads the model and returns it.
    """
    # unpack config
    model_name = model_config["model_name"]
    model_params = model_config["model_params"]
    classification_head_params = model_config["classification_head_params"]
    train_params = merge_dicts(
        model_config["default_train_params"],
        model_config[f"@{task}_override"],
    )
    data_module_params = train_params["data_module_params"]

    # load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # remove lightning module prefix
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        state_dict[key.replace("model.", "", 1)] = value
    state_dict.keys()

    # load data module
    data_module = PAN23DataModule.from_joint_config(data_module_params)

    # build model
    model = ModelBuilder.build(
        model_name,
        model_params,
        data_module.get_vocab_size(),
        data_module.get_padding_idx(),
    )

    # add model head
    model_with_class_head = ModelWithClassificationHead(
        model,
        model.get_out_embedding_dim(),
        **classification_head_params,
    )

    # load weights from checkpoint
    model_with_class_head.load_state_dict(state_dict)

    return model_with_class_head, data_module


def build_argparse():
    parser = argparse.ArgumentParser(
        description="Load a pretrained model, evaluate (F1) each document and get average"
    )

    # model_config is a file path to a JSON file
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="File path to the configuration JSON file",
    )

    # Argument for the directory of the original data
    parser.add_argument(
        "--source-data-dir",
        type=str,
        help="Path to the directory where the original data is stored.",
    )

    # checkpoint_path is a string
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )

    # task is an integer and can only take values 1, 2, or 3
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Task identifier (1, 2, or 3)",
    )

    return parser


def main():
    parser = build_argparse()
    args = parser.parse_args()

    # load config and model
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    model, data_module = load_model_and_data_module(
        model_config, args.checkpoint, f"task{args.task}"
    )
    collator_fn = data_module.loader_config["collate_fn"]

    # load dataset
    dataset = build_validation_dataset_for_task(args.task, args.source_data_dir)
    
    evaluate(model, dataset, collator_fn)


if __name__ == "__main__":
    main()
