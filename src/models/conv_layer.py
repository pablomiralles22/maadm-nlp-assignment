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
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = self.layer_norm1(x)
        x = self.ff(x)
        x = self.layer_norm2(x)
        x = x.transpose(-1, -2)
        return x