"""Pretrained speaker-verification embedding frontend + linear/MLP probe.
Method 3 in the assignment doc:
  Rathnayake & Aggarwal (?), "Domain Adaptation for Speaker Recognition in
  Singing and Spoken Voice," ICASSP 2022 — cites
  rssr25/voice-recognition-speak-sing, which studies transferring VoxCeleb
  speaker-verification embeddings (spoken voice) to singing voice.

We reuse that paper's premise directly: a speaker embedding model trained
on spoken voice (VoxCeleb) should carry useful identity information even
under the speak/sing domain gap, especially with a classifier head
fine-tuned on top for the target (singing) domain. Backbone:
`speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, Desplanques et al. 2020),
16kHz native — no resampling needed for our pipeline.

The original paper's contribution is an *adversarial domain-adaptation*
training scheme (a domain-discriminator pushing the encoder toward
domain-invariant features) on top of this embedding — not reproduced here
(flagged as a stretch goal / deep-research follow-up in
EXPERIMENT_LOG.md); we implement the paper's baseline setup (frozen/
fine-tuned pretrained speaker embedding + classifier head) faithfully.

Runs in the separate `hw1_ssl_env` conda env (torch>=2.6) alongside
ssl_frontend.py — see that file's docstring for why.
"""
import torch.nn as nn


class SpeakerEmbeddingProbe(nn.Module):
    def __init__(self, n_class=20, source="speechbrain/spkrec-ecapa-voxceleb",
                 freeze_backbone=True, hidden_dim=256):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier

        self.backbone = EncoderClassifier.from_hparams(
            source=source, savedir=f"pretrained_models/{source.split('/')[-1]}"
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.mods.parameters():
                p.requires_grad = False
            self.backbone.mods.eval()

        emb_dim = self.backbone.mods.embedding_model.emb_lin.w.out_features if hasattr(
            self.backbone.mods.embedding_model, "emb_lin"
        ) else 192  # ECAPA-TDNN default embedding size

        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_class),
        )

    def embed(self, x):
        # speechbrain's encode_batch expects (B, T) waveforms, returns (B, 1, emb_dim)
        emb = self.backbone.encode_batch(x, normalize=False)
        return emb.squeeze(1)

    def forward(self, x):
        emb = self.embed(x)
        return self.head(emb)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # a frozen backbone's requires_grad=False stops gradients, but
            # BatchNorm buffers (running_mean/var) still update on every
            # forward pass in train mode regardless of requires_grad — keep
            # it in eval mode so "frozen" is actually frozen (buffers
            # included), matching ssl_frontend.py's same guard.
            self.backbone.mods.eval()
        return self
