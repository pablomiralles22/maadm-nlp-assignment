import torch

from torch import nn

class ConvLayer(nn.Module):
    def __init__(
        self,
        conv_params: dict,
        dim_feedforward: int,
        dropout_params: dict,
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(**conv_params)
        self.ff = nn.Sequential(
            nn.Linear(conv_params["out_channels"], dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, conv_params["out_channels"]),
        )
        self.layer_norm1 = nn.LayerNorm(conv_params["out_channels"])
        self.layer_norm2 = nn.LayerNorm(conv_params["out_channels"])
        self.dropout = nn.Dropout(**dropout_params)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = self.layer_norm1(x)
        x = self.ff(x)
        x = self.layer_norm2(x)
        return x

class ConvModel(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        padding_idx: int,
        conv_layers_params: list[dict],
    ):
        super(ConvModel, self).__init__()

        embedding_dim = conv_layers_params[0]["conv_params"]["in_channels"]
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.conv_layers = nn.ModuleList(
            [
                ConvLayer(**conv_layer_param)
                for conv_layer_param in conv_layers_params
            ]
        )

    def forward(
        self,
        input_ids,  # (...BATCH_SIZE, SEQ_LEN)
    ):
        x = self.embeddings(input_ids)  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return torch.mean(x, dim=-2) # (...BATCH_SIZE, EMBEDDING_DIM)
