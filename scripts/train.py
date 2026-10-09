#!/usr/bin/env python3
"""Train a CircEnvFormer model.

This entry point provides the public command-line interface for the training
stage. Dataset-specific loading and preprocessing should be supplied by the
user for the local ERA5/IMS data.
"""
from __future__ import annotations

import argparse
import torch

from circenvformer import CircEnvEncoder, RainfallClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CircEnvFormer.")
    parser.add_argument("--checkpoint", help="Optional output checkpoint path.")
    parser.add_argument("--in-channels", type=int, default=75)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--stations", type=int, default=27)
    args = parser.parse_args()

    encoder = CircEnvEncoder(in_channels=args.in_channels, embed_dim=args.embed_dim)
    head = RainfallClassifier(embed_size=args.embed_dim, n_stations=args.stations)

    print("Model initialized.")
    print("Encoder parameters:", sum(p.numel() for p in encoder.parameters()))
    print("Forecast-head parameters:", sum(p.numel() for p in head.parameters()))

    if args.checkpoint:
        torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, args.checkpoint)
        print(f"Saved initialization checkpoint to: {args.checkpoint}")


if __name__ == "__main__":
    main()
