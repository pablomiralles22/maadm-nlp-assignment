from typing import Literal
from models.conv1d_model import Conv1dModel
from models.pretrained_transformer_model import PretrainedTransformerModel
from models.ensemble_model import EnsembleModel

Model = Literal["ensemble", "conv1d", "pretrained_transformer", "headed_model"]


class ModelBuilder:
    @classmethod
    def build(cls, name: Model, params: dict, num_embeddings: int, padding_idx: int):
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
        models = [
            cls.build(model_name, model_params, num_embeddings, padding_idx)
            for model_name, model_params in params["models"].items()
        ]
        return EnsembleModel(models)

    @classmethod
    def _build_conv1d_model(cls, params: dict, num_embeddings: int, padding_idx: int):
        return Conv1dModel(**params, num_embeddings=num_embeddings, padding_idx=padding_idx)

    @classmethod
    def _build_pretrained_transformer_model(cls, params: dict):
        return PretrainedTransformerModel(**params)
