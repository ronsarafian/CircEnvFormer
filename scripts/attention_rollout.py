#!/usr/bin/env python3
"""Compute normalized CLS attention-rollout scores from saved attention tensors."""
from __future__ import annotations

import argparse
import torch

from circenvformer.attention import compute_attention_rollout, cls_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attention rollout.")
    parser.add_argument("input", help="Torch file containing a list of attention tensors.")
    parser.add_argument("output", default="rollout.pt")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    attentions = payload["attentions"] if isinstance(payload, dict) else payload
    rollout = compute_attention_rollout(attentions)
    scores = cls_rollout(rollout)
    torch.save(scores, args.output)
    print("Rollout shape:", tuple(scores.shape))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
