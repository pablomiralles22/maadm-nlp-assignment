from abc import ABC, abstractmethod
from torch import nn
from utils.custom_types import TokenizerEncoding


class BaseModel(nn.Module, ABC):
    def forward(
        self, joint_encoding: TokenizerEncoding, disjoint_encoding: TokenizerEncoding
    ):
        """
        Returns a tensor of shape (BATCH, EMB_DIM) or (BATCH, 2*EMB_DIM) if the
        pairs of texts are encoded disjointly.
        """
        if self.does_joint_forward():
            return self._forward(
                joint_encoding.input_ids,
                joint_encoding.attention_mask,
                joint_encoding.token_type_ids,
            )

        x = self._forward(
            disjoint_encoding.input_ids,
            disjoint_encoding.attention_mask,
            disjoint_encoding.token_type_ids,
        )  # (BATCH, EMB_DIM)
        batch_dim_x_2, _ = x.shape
        return x.reshape(batch_dim_x_2 // 2, -1)

    @abstractmethod
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Should return a tensor of shape (BATCH, EMB_DIM), a single embedding for
        the whole input.
        """
        raise NotImplementedError("forward not implemented")

    def get_out_embedding_dim(self):
        """
        Returns the embedding dimension of the output of the model, times 2 if
        text pairs are encoded separately.
        """
        if self.does_joint_forward():
            return self._get_out_embedding_dim()
        return self._get_out_embedding_dim() * 2

    @abstractmethod
    def _get_out_embedding_dim(self):
        """
        Should return the embedding dimension of the output of the model.
        """
        raise NotImplementedError("get_out_embedding_dim not implemented")

    def does_joint_forward(self):
        """
        Whether the model performs a forward pass where each text in the pair
        is passed through the model separately and then concatenated.
        Defaults to false. For now, only pretrained tranformers allow processing
        pairs of texts jointly.
        """
        return False
