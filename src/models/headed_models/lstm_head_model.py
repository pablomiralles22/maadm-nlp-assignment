from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.headed_models.head_model import HeadModel

class LSTMHeadModel(HeadModel):
    def __init__(
        self,
        lstm_params: dict,
        input_size: int,
    ):
        super(LSTMHeadModel, self).__init__()
        self.lstm = nn.LSTM(
            **lstm_params,
            input_size=input_size,
            batch_first=True,
        )
        # calculate output embedding dim
        D = 2 if self.lstm.bidirectional is True else 1
        self.output_embedding_dim = D * self.lstm.hidden_size

    def get_out_embedding_dim(self):
        return self.output_embedding_dim

    def forward(
        self,
        embeddings,  # (...BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
        attention_mask,  # (...BATCH_SIZE, SEQ_LEN)
    ):
        input_lengths = attention_mask.sum(dim=-1).cpu()
        x = pack_padded_sequence(embeddings, input_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x)
        out = pad_packed_sequence(lstm_out, batch_first=True)[0]
        return out[..., -1, :]