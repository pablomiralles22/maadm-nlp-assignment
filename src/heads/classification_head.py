import torch

from torch import nn

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, dropout_p=0.1):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        self.ff = nn.Sequential(
            nn.Linear(2 * input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self, 
        input_ids,  # (2*BATCH_SIZE, MAX_LEN)
        attention_mask,  # (2*BATCH_SIZE, MAX_LEN)
    ):
        x = self.model(input_ids, attention_mask=attention_mask)  # (2*BATCH_SIZE, EMBED_DIM)
        batch_size_by_2, embed_dim = x.shape
        x = x.view(batch_size_by_2 // 2, 2 * embed_dim)  # (BATCH_SIZE, 2*EMBED_DIM)
        return torch.sigmoid(self.ff(x))
