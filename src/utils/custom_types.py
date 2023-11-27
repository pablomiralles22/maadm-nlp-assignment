from typing import Literal

EnsembleStrategy = Literal["sum", "mean", "none", "concat"]
ReductionMethod = Literal["cls", "mean", "none"]