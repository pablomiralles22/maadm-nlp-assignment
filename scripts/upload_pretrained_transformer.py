import os
import sys
import torch
import json
import argparse

from dotenv import dotenv_values

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.model_builder import ModelBuilder
from heads.classification_head import ModelWithClassificationHead
from data_loaders.pan23 import PAN23DataModule
from utils.merge_dicts import merge_dicts


def extract_pretrained_transformer_model(
    model_config: dict,
    checkpoint_path: str,
    task: str,
    hf_token: str,
    hf_repository: str,
):
    """
    Takes the checkpoint of a pytorch lightning module. If it is a model
    with a classification head, and with a base PretrainedTransformerModel,
    then it the base model is uploaded to huggingface hub.
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

    # upload to huggingface hub
    transfomer_model = model_with_class_head.model.transformer_model
    transfomer_model.push_to_hub(
        hf_repository,
        token=hf_token,
    )


def build_argparse():
    parser = argparse.ArgumentParser(
        description="Extract a pretrained transformer model and upload to huggingface hub"
    )

    # model_config is a file path to a JSON file
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="File path to the model configuration JSON file",
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

    # hf_token is a string
    parser.add_argument(
        "--hf-env-file", type=str, required=True, help="Env file with Hugging Face API token"
    )

    # hf_repository is a string
    parser.add_argument(
        "--hf-repository", type=str, required=True, help="Hugging Face repository name"
    )

    return parser

def main():
    parser = build_argparse()
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    task = f"task{args.task}"

    env_config = dotenv_values(args.hf_env_file)
    hf_token = env_config["HUGGINGFACE_WRITE_TOKEN"]

    extract_pretrained_transformer_model(
        model_config=model_config, 
        checkpoint_path=args.checkpoint,
        task=task,
        hf_token=hf_token,
        hf_repository=args.hf_repository,
    )


if __name__ == "__main__":
    main()
