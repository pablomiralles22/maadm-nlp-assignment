from utils.custom_types import ReductionMethod

def reduce(
    x,  # (...BATCH, SEQ_LEN, HIDDEN_DIM)
    reduction_method: ReductionMethod,
):
    match reduction_method:
        case "cls":
            return x[:, 0, :]
        case "mean":
            return x.mean(dim=1)
        case _:
            raise ValueError(f"Invalid reduction method: {reduction_method}")