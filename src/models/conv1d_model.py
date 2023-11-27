import torch

from torch import nn
from typing import Literal
from models.base_model import BaseModel

class Conv1dLayer(nn.Module):
    def __init__(
        self,
        conv_params: dict,
        dim_feedforward: int,
        dropout_params: dict,
    ):
        super(Conv1dLayer, self).__init__()

        self.conv = nn.Conv1d(**conv_params)
        self.ff = nn.Sequential(
            nn.Linear(conv_params["out_channels"], dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, conv_params["out_channels"]),
        )
        self.layer_norm = nn.LayerNorm(conv_params["out_channels"])
        self.dropout = nn.Dropout(**dropout_params)

    def forward(
        self,
        x,  # (...BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    ):
        x = self.conv(x)  # (...BATCH_SIZE, OUT_CHANNELS, SEQ_LEN)
        x = x.transpose(-1, -2)  # (...BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
        x = self.ff(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = x.transpose(-1, -2)  # (...BATCH_SIZE, OUT_CHANNELS, SEQ_LEN)
        return x

Conv1dModelLayer = Literal["conv", "max_pool", "mean_pool"]

class Conv1dModelLayerBuilder:
    @classmethod
    def build(cls, name: Conv1dModelLayer, params: dict):
        match name:
            case "conv":
                return Conv1dLayer(**params)
            case "max_pool":
                return nn.MaxPool1d(**params)
            case "mean_pool":
                return nn.AvgPool1d(**params)
            case _:
                raise ValueError(f"Invalid Conv1dModelLayer name: {name}")


class Conv1dModel(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        padding_idx: int,
        layer_params: list[tuple[Conv1dModelLayer, dict]],
    ):
        super(Conv1dModel, self).__init__()

        conv_layers_params = [params for layer, params in layer_params if layer == "conv"]

        assert len(conv_layers_params) > 0, "Conv1dModel must have at least one conv layer"

        # build embedding layer
        in_embedding_dim = conv_layers_params[0]["conv_params"]["in_channels"]
        self.embeddings = nn.Embedding(num_embeddings, in_embedding_dim, padding_idx)

        self.layers = nn.ModuleList(
            Conv1dModelLayerBuilder.build(name, params)
            for name, params in layer_params
        )

        # calc output embedding dim
        self.output_embedding_dim = conv_layers_params[-1]["conv_params"]["out_channels"]

    def get_out_embedding_dim(self):
        return self.output_embedding_dim

    def forward(
        self,
        input_ids,  # (...BATCH_SIZE, SEQ_LEN)
        attention_mask,  # (...BATCH_SIZE, SEQ_LEN)
        token_type_ids=None,  # (...BATCH_SIZE, SEQ_LEN)
    ):
        x = self.embeddings(input_ids)  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        x = x.transpose(-1, -2)   # (...BATCH_SIZE, EMBEDDING_DIM, SEQ_LEN)
        for layer in self.layers:
            x = layer(x)
        return torch.mean(x, dim=-1) # (...BATCH_SIZE, EMBEDDING_DIM)
