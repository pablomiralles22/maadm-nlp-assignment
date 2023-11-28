import torch

from torch import nn
from models.base_model import BaseModel

def get_model_with_classification_head(
    model: BaseModel,
    ff_dim: int,
    dropout_p=0.1,
):
    input_dim = model.get_out_embedding_dim()
    return ModelWithClassificationHead(model, input_dim, ff_dim, dropout_p)

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, dropout_p=0.1):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs)
        return torch.sigmoid(self.ff(x))
