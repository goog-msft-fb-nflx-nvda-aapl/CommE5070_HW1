"""SimCLR-style contrastive pretraining, from scratch, on the training split
only. Per the round-4 Deep Research consensus (deep_research_response_4_*.md
— Perplexity's recipe is the most concretely sourced): same-track two-crop
positives, NT-Xent loss, small projection head discarded after pretraining.
Fully from-scratch-eligible (self-supervised on our own 949 tracks, no
external data/weights). Reuses `sota_crnn`'s encoder (`CRNN.embed()`).

Realistic expectation per the round-4 responses: +1-3pp is the plausible
default, +3-6pp a strong success, >6pp unlikely — not assumed, just the
prior stated going in; see MATERIALS.md for what actually happened.

Usage:
    python -m src.pretrain_ssl --data_index_dir data/index --out_dir results/ssl_pretrain \
        --epochs 150 --batch_size 64
Then fine-tune with:
    python -m src.train --model sota_crnn --init_encoder results/ssl_pretrain/encoder.pt ...
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.ssl_dataset import TwoCropSSLDataset
from src.models.sota_cnn import CRNN


def nt_xent_loss(z_a, z_b, temperature=0.2):
    """Standard NT-Xent/InfoNCE (SimCLR). z_a, z_b: (N, D) projections for
    the two views of N tracks in the batch — 2N total, each anchor's
    positive is its paired view, negatives are all other 2(N-1) views."""
    n = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    self_mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float("-inf"))

    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_index_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.data_index_dir}/labels.json") as f:
        n_class = len(json.load(f))  # only for CRNN's constructor signature; unused head at pretrain time

    encoder = CRNN(n_class=n_class).to(args.device)
    embed_dim = encoder.layer5.hidden_size
    proj = ProjectionHead(embed_dim).to(args.device)

    ds = TwoCropSSLDataset(f"{args.data_index_dir}/train.json")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                         drop_last=True, persistent_workers=args.num_workers > 0)

    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    for epoch in range(args.epochs):
        encoder.train()
        proj.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for view_a, view_b in loader:
            view_a, view_b = view_a.to(args.device), view_b.to(args.device)
            opt.zero_grad()
            z_a = proj(encoder.embed(view_a))
            z_b = proj(encoder.embed(view_b))
            loss = nt_xent_loss(z_a, z_b, temperature=args.temperature)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"[ssl_pretrain] epoch={epoch} loss={avg_loss:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "loss": avg_loss})

    with open(os.path.join(args.out_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    encoder_path = os.path.join(args.out_dir, "encoder.pt")
    torch.save({"encoder_state": encoder.state_dict()}, encoder_path)
    print(f"done. encoder saved to {encoder_path}")


if __name__ == "__main__":
    main()
