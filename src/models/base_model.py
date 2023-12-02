from abc import ABC, abstractmethod
from torch import nn
from utils.custom_types import TokenizerEncoding


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in the project. It also abstracts the handling of
    text pairs, which can be tokenized jointly or disjointly.
    * If the model processes the pairs jointly, then each pair produces a single
        embedding in the inherited model, and everything is OK.
    * If the model processes the pairs disjointly, then each text in the pair
        produces an embedding, and the two embeddings are concatenated. This class
        also corrects the output dimensionality of the model to account for this.

    Methods:
        forward: Performs the forward pass of the model.
        _forward: Abstract method to be implemented by subclasses.
        get_out_embedding_dim: Returns the embedding dimension of the model's output.
        _get_out_embedding_dim: Abstract method to be implemented by subclasses.
        does_joint_forward: Returns whether the model performs a joint forward pass.

    """

    def forward(
        self, joint_encoding: TokenizerEncoding, disjoint_encoding: TokenizerEncoding
    ):
        """
        Returns a tensor of shape (BATCH, EMB_DIM). This method handles the concatenation
        of the two embeddings in case the model processes the text pairs disjointly.

        Args:
            joint_encoding (TokenizerEncoding): Text pairs tokenized jointly, that
                is, of size (BATCH, SEQ_LEN).
            disjoint_encoding (TokenizerEncoding): Text pairs tokenized disjointly, that
                is, of size (2*BATCH, SEQ_LEN).

        Returns:
            torch.Tensor: Tensor of shape (BATCH, EMB_DIM).

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
        )  # (BATCH*2, EMB_DIM)
        batch_dim_x_2, _ = x.shape
        return x.reshape(batch_dim_x_2 // 2, -1)

    @abstractmethod
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Should return a tensor of shape (BATCH, EMB_DIM), a single embedding for
        the whole input.

        Args:
            input_ids (torch.Tensor): Tensor of shape (BATCH, SEQ_LEN) containing the input IDs.
            attention_mask (torch.Tensor): Tensor of shape (BATCH, SEQ_LEN) containing the attention mask.
            token_type_ids (torch.Tensor, optional): Tensor of shape (BATCH, SEQ_LEN) containing the token type IDs. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape (BATCH, EMB_DIM).

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        """

        raise NotImplementedError("forward not implemented")

    def get_out_embedding_dim(self):
        """
        Returns the embedding dimension of the output of the model, times 2 if
        text pairs are encoded separately.

        Args:
            None

        Returns:
            int: The embedding dimension of the model's output.

        """

        if self.does_joint_forward():
            return self._get_out_embedding_dim()
        return self._get_out_embedding_dim() * 2

    @abstractmethod
    def _get_out_embedding_dim(self):
        """
        Should return the embedding dimension of the output of the model.

        Args:
            None

        Returns:
            int: The embedding dimension of the model's output.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        """

        raise NotImplementedError("get_out_embedding_dim not implemented")

    def does_joint_forward(self):
        """
        Whether the model performs a forward pass where each text in the pair
        is passed through the model separately and then concatenated.
        Defaults to false. For now, only pretrained tranformers allow processing
        pairs of texts jointly.

        Args:
            None

        Returns:
            bool: True if the model performs a joint forward pass, False otherwise.

        """

        return False
