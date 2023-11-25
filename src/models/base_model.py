from abc import ABC, abstractmethod
from torch import nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def get_out_embedding_dim(self):
        raise NotImplementedError("get_out_embedding_dim not implemented")

    @abstractmethod
    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("forward not implemented")