import torch

from torch import nn
from models.conv1d_model import Conv1dModelLayer, Conv1dModelLayerBuilder
from models.headed_models.head_model import HeadModel


class ConvHeadModel(HeadModel):
    def __init__(
        self,
        layer_params: list[tuple[Conv1dModelLayer, dict]],
    ):
        super(ConvHeadModel, self).__init__()

        conv_layers_params = [params for layer, params in layer_params if layer == "conv"]
        assert len(conv_layers_params) > 0, "ConvModel must have at least one conv layer"

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
        embeddings,  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        attention_mask,  # (...BATCH_SIZE, SEQ_LEN)
    ):
        x = embeddings * attention_mask.unsqueeze(-1)  # ensure 0 at padding idxs
        x = x.transpose(-1, -2)   # (...BATCH_SIZE, EMBEDDING_DIM, SEQ_LEN)
        for layer in self.layers:
            x = layer(x)
        return torch.mean(x, dim=-1) # (...BATCH_SIZE, EMBEDDING_DIM)
