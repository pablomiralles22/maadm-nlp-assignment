import argparse
import sys
import json
import os
import pytorch_lightning as pl

from copy import deepcopy

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.model_builder import ModelBuilder
from heads.classification_head import get_model_with_classification_head
from trainers.classification_trainer import ClassificationModule
from data_loaders.pan23 import PAN23DataModule
from utils.freeze_layers import freeze_layers
from utils.merge_dicts import merge_dicts


def finetune(config, initial_model, task_name):
    print(f"Starting finetuning for task {task_name}...")

    # unpack config
    model_name = config["model_name"]
    model_params = config["model_params"]

    classification_head_params = config["classification_head_params"]

    train_params = merge_dicts(
        config["default_train_params"], 
        config[f"@{task_name}_override"],
    )

    # skip task if specified
    if train_params.get("skip") is True:
        print(f"Skipping task {task_name}...")
        return

    data_module_params = train_params["data_module_params"]
    optimizer_params = train_params["optimizer_params"]
    trainer_params = train_params["trainer_params"]
    fit_params = train_params.get("fit_params") or {}

    unfrozen_layers = train_params["unfrozen_layers"]

    # load data module
    data_module = PAN23DataModule.from_joint_config(data_module_params)

    # load model with head
    if initial_model is None:
        model = ModelBuilder.build(
            model_name,
            model_params,
            data_module.get_vocab_size(),
            data_module.get_padding_idx(),
        )
    else:
        model = initial_model

    model_with_class_head = get_model_with_classification_head(
        model,
        **classification_head_params,
    )

    # freeze layers from config
    freeze_layers(model, unfrozen_layers)

    # build training module
    classification_module = ClassificationModule(
        model=model_with_class_head,
        optimizer_config=optimizer_params,
        negative_ratio=(1. / data_module.get_positive_ratio()),
    )

    # build trainer
    trainer = pl.Trainer(**trainer_params)

    # fit model
    trainer.fit(classification_module, data_module, **fit_params)

    return model

def run(config):
    finetune(deepcopy(config), None, "task1")
    # do not reuse the model from task 1, as it is mostly adapted for topic detection
    model = finetune(deepcopy(config), None, "task2")
    model = finetune(deepcopy(config), model, "task3")


###### Main ######
def parse_config(config_file_name):
    try:
        with open(config_file_name, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_name}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{config_file_name}' is not a valid JSON file.")
        return


def main():
    # Create the argparse parser
    parser = argparse.ArgumentParser(description="Parser for configuration")

    # Add arguments to the parser
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        nargs="+",
        help="Path to the JSON file containing the configuration",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the model configuration from the provided JSON file
    configs = [parse_config(config_file_name) for config_file_name in args.config]

    # create required directories
    for config in configs:
        run(config)


# Run the main function
if __name__ == "__main__":
    main()
