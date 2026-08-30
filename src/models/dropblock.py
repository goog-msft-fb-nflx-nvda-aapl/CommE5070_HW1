"""DropBlock (Ghiasi et al., NeurIPS 2018) — structured spatial dropout for
conv feature maps, zeroing contiguous blocks instead of independent units.
Qwen round-3 flagged this as "specifically validated for audio spectrograms"
without a citation; tested directly here rather than accepted or dismissed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2d(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=5):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        n, c, h, w = x.shape
        bs = min(self.block_size, h, w)
        gamma = self.drop_prob / (bs ** 2) * (h * w) / ((h - bs + 1) * (w - bs + 1))
        mask = (torch.rand(n, c, h - bs + 1, w - bs + 1, device=x.device) < gamma).float()
        mask = F.pad(mask, [bs // 2] * 4)
        mask = F.max_pool2d(mask, kernel_size=bs, stride=1, padding=bs // 2)
        mask = mask[:, :, :h, :w]
        mask = 1 - mask
        count = mask.numel()
        keep = mask.sum()
        return x * mask * (count / keep.clamp(min=1.0))
