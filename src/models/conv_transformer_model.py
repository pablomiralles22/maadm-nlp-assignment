import torch

from torch import nn
from transformers import AutoModel
from sklearn.decomposition import PCA
from models.conv_model import ConvModel

class ConvTransformer(nn.Module):
    def __init__(self, conv_layers_params: dict, transformer_model: str):
        super(ConvTransformer, self).__init__()
        self.transformer_model = AutoModel.from_pretrained(transformer_model)
        pretrained_embeddings = (
            self.transformer_model
            .embeddings
            .word_embeddings
            .weight
            .cpu().detach().numpy()
        )
        num_embeddings, transformed_embedding_dim = pretrained_embeddings.shape
        padding_idx = self.transformer_model.embeddings.word_embeddings.padding_idx

        self.conv_model = ConvModel(num_embeddings, padding_idx, conv_layers_params)

        # initialize conv model embeddings with pretrained embeddings through PCA
        conv_embedding_dim = self.conv_model.conv_layers[0].conv.in_channels
        pca = PCA(n_components=conv_embedding_dim)
        conv_init_embedding = pca.fit_transform(pretrained_embeddings)
        conv_init_embedding[padding_idx] = 0.  # so convolutions do not take these into account

        self.conv_model.embeddings.weight.data = torch.tensor(conv_init_embedding)

        # store embedding dimension
        self.output_embedding_dim = transformed_embedding_dim + self.conv_model.conv_layers[-1].conv.out_channels

    
    def forward(self, input_ids, attention_mask):
        x_transformed = self.transformer_model(input_ids, attention_mask=attention_mask).pooler_output
        x_conv = self.conv_model(input_ids)
        return torch.cat((x_transformed, x_conv), dim=-1)

    def get_output_embedding_dim(self):
        return self.output_embedding_dim
