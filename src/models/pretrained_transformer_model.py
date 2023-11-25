from typing import Literal
from transformers import AutoModel
from models.base_model import BaseModel

ReductionMethod = Literal["cls", "mean"]

class PretrainedTransformerModel(BaseModel):
    def __init__(self, transformer_model: str, transformer_reduction: ReductionMethod = "cls"):
        super(PretrainedTransformerModel, self).__init__()
        self.transformer_reduction = transformer_reduction
        self.transformer_model = AutoModel.from_pretrained(transformer_model)
        self.output_embedding_dim = self.transformer_model.embeddings.word_embeddings.weight.shape[1]

    def get_out_embedding_dim(self):
        return self.output_embedding_dim

    def forward(self, input_ids, attention_mask):
        if self.transformer_reduction == "cls":
            last_hidden_state = self.transformer_model(input_ids, attention_mask=attention_mask).last_hidden_state
            x = last_hidden_state[..., 0, :]
        elif self.transformer_reduction == "mean":
            x = self.transformer_model(input_ids, attention_mask=attention_mask).pooler_output
        else:
            raise ValueError(f"Invalid transformer_reduction: {self.transformer_reduction}")
        return x