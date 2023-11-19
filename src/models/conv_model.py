from torch import nn
from conv_layer import ConvLayer

class ConvModel(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        conv_layer_params: list[dict],
    ):
        super(ConvModel, self).__init__()

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.conv_layers = nn.ModuleList(
            [
                ConvLayer(**conv_layer_param)
                for conv_layer_param in conv_layer_params
            ]
        )

    def forward(
        self,
        input_ids,  # (...BATCH_SIZE, SEQ_LEN)
    ):
        x = self.embeddings(input_ids)  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
