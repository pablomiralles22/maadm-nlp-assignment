import torch

from torch import nn
from models.base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, models: list[BaseModel]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.out_embedding_dim = sum(model.get_out_embedding_dim() for model in models)

    def get_out_embedding_dim(self):
        return self.out_embedding_dim

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = [model.forward(input_ids, attention_mask, token_type_ids) for model in self.models]
        return torch.cat(x, dim=-1)