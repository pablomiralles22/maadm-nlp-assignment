from torch import nn

class ModelWithProjectionHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, output_dim, dropout_p):
        super(ModelWithProjectionHead, self).__init__()
        self.model = model
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask=attention_mask)
        return self.projection_head(x)

    def get_output_embedding_dim(self):
        return self.projection_head[-2].out_features
