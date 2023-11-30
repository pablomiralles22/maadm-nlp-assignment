from typing import ClassVar
from transformers import AutoModel
from models.base_model import BaseModel
from utils.custom_types import ReductionMethod
from dotenv import dotenv_values


class PretrainedTransformerModel(BaseModel):
    __TOKEN_ENV_KEY: ClassVar[str] = "HUGGINGFACE_READ_TOKEN"

    def __init__(
        self,
        transformer_model: str,
        transformer_reduction: ReductionMethod = "cls",
        joint_forward: bool = True,
        token_env_file: str = None,
        **kwargs,
    ):
        super(PretrainedTransformerModel, self).__init__()
        self.transformer_reduction = transformer_reduction

        token = None
        if token_env_file is not None:
            env_config = dotenv_values(token_env_file)
            token = env_config[self.__TOKEN_ENV_KEY]

        self.transformer_model = AutoModel.from_pretrained(transformer_model, **kwargs, token=token)
        self.output_embedding_dim = (
            self.transformer_model.embeddings.word_embeddings.weight.shape[1]
        )
        self.joint_forward = joint_forward

    def _get_out_embedding_dim(self):
        return self.output_embedding_dim

    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.transformer_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self._reduce(out)

    def _reduce(self, out):
        if self.transformer_reduction == "cls":
            last_hidden_state = out.last_hidden_state
            return last_hidden_state[..., 0, :]
        elif self.transformer_reduction == "mean":
            return out.pooler_output
        elif self.transformer_reduction == "none":
            return out.last_hidden_state
        else:
            raise ValueError(
                f"Invalid transformer_reduction: {self.transformer_reduction}"
            )

    def does_joint_forward(self):
        return self.joint_forward
