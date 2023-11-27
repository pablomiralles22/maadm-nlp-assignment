from transformers import AutoModel
from models.base_model import BaseModel
from utils.custom_types import ReductionMethod

class PretrainedTransformerModel(BaseModel):
    def __init__(self, transformer_model: str, transformer_reduction: ReductionMethod = "cls", **kwargs):
        super(PretrainedTransformerModel, self).__init__()
        self.transformer_reduction = transformer_reduction
        self.transformer_model = AutoModel.from_pretrained(transformer_model, **kwargs)
        self.output_embedding_dim = self.transformer_model.embeddings.word_embeddings.weight.shape[1]

    def get_out_embedding_dim(self):
        return self.output_embedding_dim

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.transformer_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.transformer_reduction == "cls":
            last_hidden_state = out.last_hidden_state
            x = last_hidden_state[..., 0, :]
        elif self.transformer_reduction == "mean":
            x = out.pooler_output
        elif self.transformer_reduction == "none":
            x = out.last_hidden_state
        else:
            raise ValueError(f"Invalid transformer_reduction: {self.transformer_reduction}")
        return x
