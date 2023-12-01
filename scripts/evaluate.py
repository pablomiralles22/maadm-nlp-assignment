import os
import sys
import torch
import json
import argparse
import re
import numpy as np
import torchmetrics
from transformers import AutoTokenizer, AutoModel

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.model_builder import ModelBuilder
from heads.classification_head import ModelWithClassificationHead
from data_loaders.pan23 import PAN23DataModule, PAN23CollatorFn
from utils.merge_dicts import merge_dicts
#from pan-data-transform import files_to_dicts # error


def predict(model, batch):
    model.eval()

    labels = batch["labels"].float().view(-1, 1)

    logits = model.forward(batch["joint_encoding"].input_ids, batch["joint_encoding"].attention_mask, batch["joint_encoding"].token_type_ids)
    #logits = model.forward(batch["joint_encoding"], batch["disjoint_encoding"])
    #logits = model(batch["joint_encoding"].input_ids, batch["joint_encoding"].attention_mask, batch["joint_encoding"].token_type_ids)

    print(logits.pooler_output.shape) # tengo q añadrile la cabeza al modelo??
    logits = torch.nn.functional.softmax(logits.pooler_output, dim=1)

    return logits, labels

def get_positive_ratio(truths):
    cnt = 0
    for label in truths:
        if label == 1:
            cnt += 1
    return cnt / len(truths)

def files_to_dicts(source_path):
    filenames = os.listdir(source_path)

    lines_by_id = dict()
    truths_by_id = dict()

    for filename in filenames:
        is_truth = filename.startswith("truth")
        problem_id = re.search(r"\d+", filename).group(0)
        filepath = os.path.join(source_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            if is_truth is True:
                truths_by_id[problem_id] = json.load(f)["changes"]
            else:
                lines_by_id[problem_id] = f.readlines()

    return lines_by_id, truths_by_id

def load_pretrained_classification_model(
    model_config: dict,
    checkpoint_path: str,
    task: str
):
    """
    Takes the checkpoint of a pytorch lightning module. It must be a model
    with a classification head. Then, it evaluates model for each document
    in desired task.
    """
    # unpack config
    model_name = model_config["model_name"]
    model_params = model_config["model_params"]
    train_params = merge_dicts(
        model_config["default_train_params"],
        model_config[f"@{task}_override"],
    )
    data_module_params = train_params["data_module_params"]

    # load pycheckpoint
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
        dropout_p=0.25,
        ff_dim=2048,
    )

    # load weights from checkpoint
    model_with_class_head.load_state_dict(state_dict)

    return model_with_class_head


def build_argparse():
    parser = argparse.ArgumentParser(
        description="Load a pretrained model, evaluate (F1) each document and get average"
    )

    # model_config is a file path to a JSON file
    parser.add_argument(
        "--config",
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

    task = f"task{args.task}"
    set_type = "validation"
    source_data_path = os.path.join(
                args.source_data_dir,
                f"pan23-multi-author-analysis-dataset{args.task}/pan23-multi-author-analysis-dataset{args.task}-{set_type}/",
            )
    
    # load model
    with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    model_params = config["model_params"]["models"]
    train_params = config["default_train_params"]

    model = AutoModel.from_pretrained(model_params['pretrained_transformer']['transformer_model'])
    tokenizer = AutoTokenizer.from_pretrained(train_params['data_module_params']['tokenizer'])

    f1_score = torchmetrics.F1Score(task="binary")
    max_len = train_params['data_module_params']['max_len']

    lines_by_id, truths_by_id = files_to_dicts(source_data_path)
    collator_fn = PAN23CollatorFn(tokenizer, max_len)

    f1_losses = {}
    for problem_id, lines in lines_by_id.items():
        batch = []
        truths = truths_by_id[problem_id]  
        for text_1, text_2, label in zip(lines, lines[1:], truths):
            batch.append({"text1": text_1, "text2": text_2, "label": label})

        batch = collator_fn(batch)
        logits, labels = predict(model, batch)

        score_f1 = f1_score(logits, labels)
        f1_losses[problem_id] = score_f1
        break
    print(f1_losses)


if __name__ == "__main__":
    main()
