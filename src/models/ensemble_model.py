import torch

from torch import nn
from utils.custom_types import TokenizerEncoding
from models.base_model import BaseModel

class EnsembleModel(BaseModel):
    """
    Ensemble of models. The forward pass of each model is performed and the
    outputs are concatenated.
    WARNING: we assume this is a top level model, i.e. it is not used as a
    submodel of another ensemble.
    """
    def __init__(self, models: list[BaseModel]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.out_embedding_dim = sum([model.get_out_embedding_dim() for model in self.models])

    def get_out_embedding_dim(self):
        return self.out_embedding_dim

    def forward(
        self, joint_encoding: TokenizerEncoding, disjoint_encoding: TokenizerEncoding
    ):
        x = [model.forward(joint_encoding, disjoint_encoding) for model in self.models]
        return torch.cat(x, dim=-1)

    def _get_out_embedding_dim(self):
        pass
    
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        pass

