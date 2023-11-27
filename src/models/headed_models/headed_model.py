import torch

from models.base_model import BaseModel
from models.headed_models.head_model import HeadModel
from utils.custom_types import EnsembleStrategy, ReductionMethod
from utils.reduce import reduce


class HeadedModel(BaseModel):
    def __init__(
        self,
        base_model: BaseModel,
        head_model: HeadModel,
        base_model_reduction: ReductionMethod = "cls",
        ensemble_strategy: EnsembleStrategy = "concat",
    ):
        super(HeadedModel, self).__init__()
        self.base_model = base_model
        self.head_model = head_model
        self.base_model_reduction = base_model_reduction
        self.ensemble_strategy = ensemble_strategy

    def get_out_embedding_dim(self):
        if self.ensemble_strategy == "concat":
            return self.base_model.get_out_embedding_dim() + self.head_model.get_out_embedding_dim()
        return self.head_model.get_out_embedding_dim()

    def forward(self, input_ids, attention_mask):
        base_model_out = self.base_model(input_ids, attention_mask=attention_mask)
        head_model_out = self.head_model(base_model_out, attention_mask=attention_mask)
        reduced_base_model_out = reduce(base_model_out, self.base_model_reduction)
        if self.ensemble_strategy == "concat":
            return torch.cat([reduced_base_model_out, head_model_out], dim=-1)
        elif self.ensemble_strategy == "sum":
            return reduced_base_model_out + head_model_out
        elif self.ensemble_strategy == "mean":
            return (reduced_base_model_out + head_model_out) / 2
        return head_model_out
