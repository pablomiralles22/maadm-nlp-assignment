from typing import Literal
from models.base_model import BaseModel
from models.headed_models.lstm_head_model import LSTMHeadModel
from models.headed_models.headed_model import HeadedModel
from utils.custom_types import EnsembleStrategy, ReductionMethod

HeadModel = Literal["lstm"]


class HeadedModelBuilder:
    @classmethod
    def build(
        cls,
        base_model: BaseModel,
        head_model_name: HeadModel,
        head_model_params: dict,
        reduction_method: ReductionMethod,
        ensemble_strategy: EnsembleStrategy,
    ):
        base_model_dim = base_model.get_out_embedding_dim()
        match head_model_name:
            case "lstm":
                head_model = cls._build_lstm_head(head_model_params, base_model_dim)
            case _:
                raise ValueError(f"Invalid head model name: {head_model_name}")
        return HeadedModel(base_model, head_model, reduction_method, ensemble_strategy)

    @classmethod
    def _build_lstm_head(cls, params: dict, input_size: int):
        return LSTMHeadModel(params, input_size=input_size)
