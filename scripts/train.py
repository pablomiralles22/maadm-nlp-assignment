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
from trainers.contrastive_pretrainer import ContrastivePretrainer
from data_loaders.blogposts import BlogDataset, BlogCollatorFn
from data_loaders.pan23 import PAN23Dataset, PAN23CollatorFn


def run(config):
    # unpack config
    log_file_name = os.path.join(config["out_dir"], f"{config['prefix_file_name']}.log")
    pretrained_model_file_name = os.path.join(
        config["out_dir"], f"{config['prefix_file_name']}_pretrained.pt"
    )
    finetuned_model_file_name = os.path.join(
        config["out_dir"], f"{config['prefix_file_name']}_finetuned.pt"
    )
    device = torch.device(config.get("device") if torch.cuda.is_available() else "cpu")
    model_params = config["model_params"]
    transformer_pretrained_model_name = model_params["transformer_model"]

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
    tokenizer = AutoTokenizer.from_pretrained(transformer_pretrained_model_name)

    # build contrastive pretrainer
    contrastive_pretrainer_config = {
        **config["pretrain_params"],
        "collator_fn": BlogCollatorFn(tokenizer, config["max_len"]),
        "log_file": log_file_name,
        "checkpoint_file": pretrained_model_file_name,
        "device": device,
    }

    dataset = BlogDataset(config["pretrain_dataset_root_dir"])
    test_size = int(contrastive_pretrainer_config["test_set_ratio"] * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
    )

    pretrainer = ContrastivePretrainer(
        contrastive_pretrainer_config, model_with_proj_head, train_dataset, test_dataset
    )

    # freeze layers selected from config
    for param in model.transformer_model.parameters():
        param.requires_grad = False

    layers = model.transformer_model.encoder.layer
    frozen_layers = len(layers) - contrastive_pretrainer_config["unfrozen_layers"]
    for layer in layers[frozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # run pretrainer
    pretrainer.run()


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
        os.makedirs(config["out_dir"], exist_ok=True)
        run(config)


# Run the main function
if __name__ == "__main__":
    main()
