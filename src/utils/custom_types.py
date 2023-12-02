import torch

from dataclasses import dataclass
from typing import Literal, Optional

EnsembleStrategy = Literal["sum", "mean", "none", "concat"]
ReductionMethod = Literal["cls", "mean", "none"]

@dataclass
class TokenizerEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
