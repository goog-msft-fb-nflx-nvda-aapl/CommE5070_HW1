"""ArcFace / AAM-Softmax margin head (Deng et al., "ArcFace: Additive Angular
Margin Loss for Deep Face Recognition," CVPR 2019), per
deep_research/round5_prior_year_gap_and_latest_literature/response_gemini.md
— recommended as a from-scratch-eligible way to make the embedding
production/album-invariant (the exact confound Hsieh et al.'s album-level
split targets), by compressing intra-class embeddings onto a hyperspherical
manifold with an explicit angular margin, rather than letting plain
cross-entropy find any separating hyperplane (including one that partly
relies on album-specific production artifacts).

Margin/scale defaults (m=0.2, s=24) are smaller than the m=0.5/s=64 typical
in large-scale (million-identity) face/speaker recognition, since this
project has only 20 classes and ~950 training tracks — a large margin on
this little data risks non-convergence. Not sourced from a specific paper's
ablation on a dataset this small (round-5's response didn't give a citable
source for its own suggested m/s either) — treat as a reasonable starting
point to test, not a validated setting.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, n_class, s=24.0, m=0.2):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(n_class, in_features) * 0.01)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embed, labels=None):
        cosine = F.linear(F.normalize(embed), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        # numerical safety for angles beyond (pi - m): fall back to a linear margin
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


class SotaCRNNArcFace(nn.Module):
    """`sota_crnn`-family encoder (`CRNN.embed()`, reused as-is) with an
    ArcFace margin head instead of a plain linear classifier."""

    requires_labels_in_forward = True

    def __init__(self, n_class=20, channel_mult=1.5, s=24.0, m=0.2):
        super().__init__()
        from src.models.sota_cnn import CRNN

        self.encoder = CRNN(n_class=n_class, channel_mult=channel_mult)
        embed_dim = self.encoder.layer5.hidden_size
        self.arc_head = ArcMarginProduct(embed_dim, n_class, s=s, m=m)
        del self.encoder.dense  # unused — ArcMarginProduct replaces the linear classifier

    def embed(self, x):
        return self.encoder.embed(x)

    def forward(self, x, labels=None):
        emb = self.embed(x)
        return self.arc_head(emb, labels)
