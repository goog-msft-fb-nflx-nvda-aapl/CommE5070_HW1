"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020), used as an
auxiliary loss alongside cross-entropy — per Qwen/Perplexity's round-3
recommendation (Tier 1 in Qwen's response, "same encoder, multiple heads"
spirit in Perplexity's): pulls same-artist embeddings together and pushes
different-artist embeddings apart directly, on top of (not instead of) the
classification loss. Fully from-scratch (uses in-batch labels only, no
pretrained weights or external data).
"""
import torch
import torch.nn.functional as F


def supcon_loss(embeddings, labels, temperature=0.1):
    """embeddings: (N, D) raw (unnormalized) embeddings from model.embed().
    labels: (N,) integer class labels. Batches with <2 examples of any given
    label contribute zero gradient for that anchor (no positives)."""
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    n = z.size(0)

    sim = torch.matmul(z, z.T) / temperature  # (N, N)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # numerical stability

    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T).float().to(device)  # (N, N)
    self_mask = torch.eye(n, device=device)
    positive_mask = same_label * (1 - self_mask)

    exp_sim = torch.exp(sim) * (1 - self_mask)  # exclude self from denominator
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = positive_mask.sum(dim=1)
    has_positive = pos_counts > 0
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[has_positive] / pos_counts[has_positive]
    return -mean_log_prob_pos.mean()
