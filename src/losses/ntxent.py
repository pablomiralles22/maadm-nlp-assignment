import torch

from torch.nn import functional as F

def ntxent_loss(emb_1, emb_2, temperature=0.07):
    batch_size, _ = emb_1.shape
    norm_emb_1, norm_emb_2 = F.normalize(emb_1), F.normalize(emb_2)
    cos_sim = torch.einsum("ax,bx->ab", norm_emb_1, norm_emb_2)
    scaled_cos_sim = cos_sim / temperature

    device = scaled_cos_sim.device
    labels = torch.arange(batch_size).to(device)
    return 0.5 * F.cross_entropy(scaled_cos_sim, labels) + 0.5 * F.cross_entropy(scaled_cos_sim.T, labels)
