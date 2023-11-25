import argparse
import sys
import json
import os
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from copy import deepcopy

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.conv_transformer_model import ConvTransformer
from heads.projection_head import ModelWithProjectionHead
from heads.classification_head import ModelWithClassificationHead
from trainers.contrastive_pretrainer import ContrastivePretrainingModule
from trainers.classification_trainer import ClassificationModule
from data_loaders.blogposts import BlogDataModule
from data_loaders.pan23 import PAN23DataModule
from utils.freeze_layers import freeze_layers



def pretrain(config):
    print("Starting pretraining...")
    # unpack config
    model_params = config["model_params"]
    pretrain_params = config["pretrain_params"]

    data_module_params = pretrain_params["data_module_params"]
    optimizer_params = pretrain_params["optimizer_params"]
    trainer_params = pretrain_params["trainer_params"]
    fit_params = pretrain_params.get("fit_params") or {}

    unfrozen_layers = pretrain_params["unfrozen_layers"]

    # load model with head
    model = ConvTransformer(**model_params)
    model_with_proj_head = ModelWithProjectionHead(
        model,
        model.output_embedding_dim,
        **config["projection_head_params"],
    )

    # freeze layers from config
    freeze_layers(model.transformer_model, unfrozen_layers)

    # load data module
    data_module = BlogDataModule.from_joint_config(data_module_params)

    # build training module
    pretraining_module = ContrastivePretrainingModule(
        model=model_with_proj_head,
        optimizer_config=optimizer_params,
    )

    # build trainer
    callbacks = [
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]
    trainer = pl.Trainer(**trainer_params, callbacks=callbacks)

    # fit model
    trainer.fit(pretraining_module, data_module, **fit_params)

    return model


def finetune(config, pretrained_model, task_name):
    print(f"Starting finetuning for task {task_name}...")

    # unpack config
    classification_head_params = config["classification_head_params"]
    pan_train_params = config["pan_train_params"]

    data_module_params = pan_train_params["data_module_params"]
    optimizer_params = pan_train_params["optimizer_params"]
    trainer_params = pan_train_params["trainer_params"]
    fit_params = pan_train_params.get("fit_params") or {}

    unfrozen_layers = pan_train_params["unfrozen_layers"]

    # load model with head
    model_with_class_head = ModelWithClassificationHead(
        pretrained_model,
        pretrained_model.output_embedding_dim,
        **classification_head_params,
    )

    # freeze layers from config
    freeze_layers(pretrained_model.transformer_model, unfrozen_layers)

    # load data module
    data_path = data_module_params["data_path"]
    task_data_path = os.path.join(data_path, task_name)
    data_module_params["data_path"] = task_data_path

    data_module = PAN23DataModule.from_joint_config(data_module_params)

    # build training module
    classification_module = ClassificationModule(
        model=model_with_class_head,
        optimizer_config=optimizer_params,
        negative_ratio=(1. / data_module.get_positive_ratio()),
    )

    # change root dir to task one
    default_root_dir = trainer_params["default_root_dir"]
    trainer_params["default_root_dir"] = os.path.join(default_root_dir, task_name)
    # build trainer
    callbacks = [
        ModelCheckpoint(
            filename="{epoch}-{val_f1_score:.2f}",
            monitor="val_f1_score",
            mode="max",
        ),
        EarlyStopping(
            monitor="val_f1_score",
            patience=10,
            mode="max",
        ),
    ]
    trainer = pl.Trainer(**trainer_params, callbacks=callbacks)

    # fit model
    trainer.fit(classification_module, data_module, **fit_params)

    return model_with_class_head


PRETRAINED_STORE_PATH = "/tmp/pretrained_model.pt"

def run(config):
    pretrained_model = pretrain(deepcopy(config))
    torch.save(pretrained_model, PRETRAINED_STORE_PATH)  # to start again for each task
    finetune(deepcopy(config), torch.load(PRETRAINED_STORE_PATH), "task1")
    finetune(deepcopy(config), torch.load(PRETRAINED_STORE_PATH), "task2")
    finetune(deepcopy(config), torch.load(PRETRAINED_STORE_PATH), "task3")


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
