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
        # the output dimension is the sum of the output dimensions of the models
        self.out_embedding_dim = sum([model.get_out_embedding_dim() for model in self.models])

    def forward(
        self, joint_encoding: TokenizerEncoding, disjoint_encoding: TokenizerEncoding
    ):
        """
        Perform the forward pass of the ensemble model. Notice that in this case we
        had to override the `forward` method of the base class, because we need to
        pass the two encodings to each model. This is because each model may process
        the encodings differently.

        Args:
            joint_encoding (TokenizerEncoding): The joint encoding.
            disjoint_encoding (TokenizerEncoding): The disjoint encoding.

        Returns:
            torch.Tensor: The concatenated outputs of the models.
        """
        x = [model.forward(joint_encoding, disjoint_encoding) for model in self.models]
        return torch.cat(x, dim=-1)

    def get_out_embedding_dim(self):
        """
        Get the output embedding dimension of the ensemble model. Notice that we had to
        override the `get_out_embedding_dim` method of the base class.

        Returns:
            int: The output embedding dimension.
        """
        return self.out_embedding_dim

    def _get_out_embedding_dim(self):
        """
        Ignore in this class, will not be used.
        """
        pass
    
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Ignore in this class, will not be used.
        """
        pass

