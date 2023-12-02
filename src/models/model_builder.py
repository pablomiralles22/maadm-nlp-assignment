from typing import Literal
from models.conv1d_model import Conv1dModel
from models.pretrained_transformer_model import PretrainedTransformerModel
from models.ensemble_model import EnsembleModel

Model = Literal["ensemble", "conv1d", "pretrained_transformer"]


class ModelBuilder:
    """A class responsible for building different models based on the given name and parameters."""

    @classmethod
    def build(cls, name: Model, params: dict, num_embeddings: int, padding_idx: int):
        """
        Build a model based on the given name and parameters.

        Args:
            name (Model): The name of the model to build.
            params (dict): The parameters for building the model.
            num_embeddings (int): The number of embeddings.
            padding_idx (int): The padding index.

        These last two parameters are used in case we need to build the embedding layer.

        Returns:
            The built model.

        Raises:
            ValueError: If an invalid model name is provided.
        """
        match name:
            case "ensemble":
                return cls._build_ensemble_model(params, num_embeddings, padding_idx)
            case "conv1d":
                return cls._build_conv1d_model(params, num_embeddings, padding_idx)
            case "pretrained_transformer":
                return cls._build_pretrained_transformer_model(params)
            case _:
                raise ValueError(f"Invalid model name: {name}")

    @classmethod
    def _build_ensemble_model(cls, params: dict, num_embeddings: int, padding_idx: int):
        """
        Build an ensemble model based on the given parameters.

        Args:
            params (dict): The parameters for building the ensemble model.
            num_embeddings (int): The number of embeddings.
            padding_idx (int): The padding index.

        Returns:
            The built ensemble model.
        """
        models = [
            cls.build(model_name, model_params, num_embeddings, padding_idx)
            for model_name, model_params in params["models"].items()
        ]
        return EnsembleModel(models)

    @classmethod
    def _build_conv1d_model(cls, params: dict, num_embeddings: int, padding_idx: int):
        """
        Build a Conv1D model based on the given parameters.

        Args:
            params (dict): The parameters for building the Conv1D model.
            num_embeddings (int): The number of embeddings.
            padding_idx (int): The padding index.

        Returns:
            The built Conv1D model.
        """
        return Conv1dModel(**params, num_embeddings=num_embeddings, padding_idx=padding_idx)

    @classmethod
    def _build_pretrained_transformer_model(cls, params: dict):
        """
        Build a pretrained transformer model based on the given parameters.

        Args:
            params (dict): The parameters for building the pretrained transformer model.

        Returns:
            The built pretrained transformer model.
        """
        return PretrainedTransformerModel(**params)
