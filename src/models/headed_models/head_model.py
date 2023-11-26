from torch import nn
from abc import ABC, abstractmethod


class HeadModel(nn.Module, ABC):
    @abstractmethod
    def get_out_embedding_dim(self):
        raise NotImplementedError(
            "HeadedModel subclasses must implement get_out_embedding_dim"
        )

    @abstractmethod
    def forward(self, embeddings, attention_mask):
        raise NotImplementedError("HeadedModel subclasses must implement forward")
