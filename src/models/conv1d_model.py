import torch

from torch import nn
from typing import Literal
from models.base_model import BaseModel

class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ff_dim: int,
    ):
        super(FeedForwardLayer, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, out_dim),
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x,  # (...BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    ):
        x = x.transpose(-1, -2)  # (...BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
        x = self.ff(x)
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)  # (...BATCH_SIZE, OUT_CHANNELS, SEQ_LEN)
        return x

Conv1dModelLayer = Literal["conv", "relu", "max_pool", "mean_pool", "dropout", "batch_norm", "ff"]

class Conv1dModelLayerBuilder:
    @classmethod
    def build(cls, name: Conv1dModelLayer, params: dict):
        match name:
            case "conv":
                return nn.Conv1d(**params)
            case "max_pool":
                return nn.MaxPool1d(**params)
            case "mean_pool":
                return nn.AvgPool1d(**params)
            case "relu":
                return nn.ReLU()
            case "dropout":
                return nn.Dropout(**params)
            case "batch_norm":
                return nn.BatchNorm1d(**params)
            case "ff":
                return FeedForwardLayer(**params)
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
        in_embedding_dim = conv_layers_params[0]["in_channels"]
        # in_embedding_dim = conv_layers_params[0]["conv_params"]["in_channels"]
        self.embeddings = nn.Embedding(num_embeddings, in_embedding_dim, padding_idx)

        self.layers = nn.ModuleList(
            Conv1dModelLayerBuilder.build(name, params)
            for name, params in layer_params
        )

        # calc output embedding dim
        self.output_embedding_dim = conv_layers_params[-1]["out_channels"]
        # self.output_embedding_dim = conv_layers_params[-1]["conv_params"]["out_channels"]

    def _get_out_embedding_dim(self):
        return self.output_embedding_dim

    def _forward(
        self,
        input_ids,  # (...BATCH_SIZE, SEQ_LEN)
        attention_mask,  # (...BATCH_SIZE, SEQ_LEN)
        token_type_ids=None,  # (...BATCH_SIZE, SEQ_LEN)
    ):  # (...BATCH_SIZE, EMBEDDING_DIM)
        x = self.embeddings(input_ids)  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        x = x.transpose(-1, -2)   # (...BATCH_SIZE, EMBEDDING_DIM, SEQ_LEN)
        for layer in self.layers:
            x = layer(x)
        return torch.mean(x, dim=-1) # (...BATCH_SIZE, EMBEDDING_DIM)
