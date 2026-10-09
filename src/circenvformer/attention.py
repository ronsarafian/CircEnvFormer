"""Attention-rollout utilities used for interpretation."""

from __future__ import annotations

import numpy as np
import torch


def compute_attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """Compute attention rollout with residual connections.

    Parameters
    ----------
    attentions
        List of tensors shaped ``[batch, heads, tokens, tokens]``.
    """
    if not attentions:
        raise ValueError("No attention maps supplied.")

    rollout = None
    for attn in attentions:
        a = attn.mean(dim=1)
        eye = torch.eye(a.size(-1), device=a.device).expand(a.size(0), -1, -1)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else torch.bmm(a, rollout)
    return rollout


def cls_rollout(rollout: torch.Tensor) -> torch.Tensor:
    """Return normalized CLS-to-token rollout scores excluding CLS itself."""
    scores = rollout[:, 0, 1:]
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
