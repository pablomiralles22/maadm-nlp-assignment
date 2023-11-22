import argparse
import torch
import sys
import json
import os

from transformers import AutoTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.conv_transformer_model import ConvTransformer
from heads.projection_head import ModelWithProjectionHead
from heads.classification_head import ModelWithClassificationHead
from trainers.contrastive_pretrainer import ContrastivePretrainer
from trainers.classification_trainer import ClassificationTrainer
from data_loaders.blogposts import BlogDataset, BlogCollatorFn
from data_loaders.pan23 import PAN23Dataset, PAN23CollatorFn

def freeze_layers(transformer_model, num_unfrozen_layers):
    for param in transformer_model.parameters():
        param.requires_grad = False

    layers = transformer_model.encoder.layer
    frozen_layers = len(layers) - num_unfrozen_layers
    for layer in layers[frozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def split_torch_dataset(dataset, test_set_ratio):
    test_size = int(test_set_ratio * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
    )
    return train_dataset, test_dataset

def pretrain(config):
    pretrained_model_path = config["pretrained_model_path"] 
    if os.path.exists(pretrained_model_path) is True:
        # do not train, just return pretrained base model
        model_with_proj_head = torch.load(pretrained_model_path)
        print("Pretrained model loaded from file!")
        return model_with_proj_head.model
    
    print("Starting pretraining...")

    log_file_name = config["log_file"]
    device = torch.device(config.get("device") if torch.cuda.is_available() else "cpu")
    model_params = config["model_params"]

    # load model
    model = ConvTransformer(
        model_params["conv_layers_params"], model_params["transformer_model"]
    )
    model_with_proj_head = ModelWithProjectionHead(
        model,
        model.output_embedding_dim,
        **model_params["projection_head_params"],
    )

    # get tokenizer for data loaders
    tokenizer = AutoTokenizer.from_pretrained(model_params["transformer_model"])

    # build contrastive pretrainer
    contrastive_pretrainer_config = {
        **config["pretrain_params"],
        "collator_fn": BlogCollatorFn(tokenizer, config["max_len"]),
        "log_file": log_file_name,
        "checkpoint_file": pretrained_model_path,
        "device": device,
    }

    train_dataset, test_dataset = split_torch_dataset(
        BlogDataset(config["pretrain_dataset_root_dir"]),
        contrastive_pretrainer_config["test_set_ratio"],
    )

    pretrainer = ContrastivePretrainer(
        contrastive_pretrainer_config, model_with_proj_head, train_dataset, test_dataset
    )

    # freeze layers selected from config
    freeze_layers(model.transformer_model, contrastive_pretrainer_config["unfrozen_layers"])

    # run pretrainer
    pretrainer.run()

    return model

def finetune(config, pretrained_model, task_name):
    print(f"Starting finetuning for task {task_name}...")

    log_file_name = config["log_file"]
    device = torch.device(config.get("device") if torch.cuda.is_available() else "cpu")
    model_params = config["model_params"]

    # load model
    model_with_class_head = ModelWithClassificationHead(
        pretrained_model,
        pretrained_model.output_embedding_dim,
        **model_params["classification_head_params"],
    )

    # get tokenizer for data loaders
    tokenizer = AutoTokenizer.from_pretrained(model_params["transformer_model"])


    # load datasets
    train_dataset = PAN23Dataset(os.path.join(config["task_dataset_root_dir"], f"pan23-{task_name}-train"))
    test_dataset = PAN23Dataset(os.path.join(config["task_dataset_root_dir"], f"pan23-{task_name}-validation"))

    # build contrastive pretrainer
    trainer_config = {
        **config["pan_train_params"],
        "collator_fn": PAN23CollatorFn(tokenizer, config["max_len"]),
        "log_file": log_file_name,
        "checkpoint_file": config["finetuned_model_path_fmt"].format(task_name),
        "device": device,
    }
    trainer = ClassificationTrainer(trainer_config, model_with_class_head, train_dataset, test_dataset)

    # freeze layers selected from config
    freeze_layers(pretrained_model.transformer_model, trainer_config["unfrozen_layers"])

    # run pretrainer
    trainer.run()

    return pretrained_model



def run(config):
    pretrained_model = pretrain(config)
    finetune(config, pretrained_model, "task1")
    finetune(config, pretrained_model, "task2")
    finetune(config, pretrained_model, "task3")

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
        os.makedirs(os.path.dirname(config["log_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(config["pretrained_model_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(config["finetuned_model_path_fmt"]), exist_ok=True)
        run(config)


# Run the main function
if __name__ == "__main__":
    main()
