"""Pretrained self-supervised (SSL) audio frontend + linear/MLP probe.

Covers two items in the assignment doc at once:
- Method 2-4: Singer identity representation learning using self-supervised
  techniques (Ma et al., ISMIR 2023, arXiv:2306.12714) — the paper studies
  exactly this recipe (frozen SSL features + lightweight classifier) for
  singer identification.
- Baseline 1-1/1-2: "pre-trained self-supervised frontends for singing voice
  understanding" (APSIPA 2023) / "speech SSL for music" (ISMIR 2023).

Default backbone: `m-a-p/MERT-v1-95M` (Li et al., "MERT: Acoustic Music
Understanding Model with Large-Scale Self-Supervised Training") — a
*music*-domain SSL model (vs. speech-domain HuBERT/wav2vec2), expected to
transfer better to singer identity than a speech-only SSL model. Requires
`trust_remote_code=True` (custom modeling code shipped in the HF repo) and
16kHz->24kHz resampling (MERT's native rate).

Note: `m-a-p/MERT-v1-95M`'s checkpoint predates HF's `safetensors`-only
`torch.load` policy for newer `transformers` — loading it requires
`torch>=2.6`. This repo's main training env pins `torch==2.5.1+cu121` (set
up before this restriction was hit) and has several concurrent jobs running
in it; rather than risk destabilizing those, this model (and
speaker_frontend.py) run in a separate `hw1_ssl_env` conda env
(`torch>=2.6+cu121`) on our training server. See EXPERIMENT_LOG.md.
"""
import torch
import torch.nn as nn
import torchaudio


class SSLLinearProbe(nn.Module):
    def __init__(self, n_class=20, backbone_name="m-a-p/MERT-v1-95M", freeze_backbone=True,
                 input_sr=16000, backbone_sr=24000, hidden_dim=256):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.input_sr = input_sr
        self.backbone_sr = backbone_sr
        self.resample = (
            torchaudio.transforms.Resample(input_sr, backbone_sr) if input_sr != backbone_sr else nn.Identity()
        )

        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_class),
        )

    def embed(self, x):
        x = self.resample(x)
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(x, output_hidden_states=False).last_hidden_state  # (B, T, H)
        return out.mean(dim=1)

    def forward(self, x):
        emb = self.embed(x)
        return self.head(emb)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep frozen backbone's BN/dropout in eval mode
        return self
