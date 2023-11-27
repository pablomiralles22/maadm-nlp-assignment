import torch

from torch import nn

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, dropout_p=0.1, joint_pairs=False):
        super(ModelWithClassificationHead, self).__init__()
        self.joint_pairs = joint_pairs

        if joint_pairs is False:
            input_dim *= 2

        self.model = model
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self, 
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        x = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.joint_pairs is False:  # reshape
            batch_size_by_2, embed_dim = x.shape
            x = x.reshape(batch_size_by_2 // 2, 2 * embed_dim)

        return torch.sigmoid(self.ff(x))
