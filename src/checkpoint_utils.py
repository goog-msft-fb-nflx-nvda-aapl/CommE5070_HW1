"""Shared checkpoint loading, handling both full (`model_state`, from
src/train.py) and head-only (`head_state`, from src/strip_checkpoint.py —
used for frozen-backbone models: the backbone is already freshly loaded
from HuggingFace/speechbrain inside the model constructor, so only the
trainable head needs restoring) checkpoints transparently.
"""
import torch


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "head_state" in ckpt:
        model.load_state_dict(ckpt["head_state"], strict=False)
    else:
        raise ValueError(f"unrecognized checkpoint format: keys={list(ckpt.keys())}")
    return model
