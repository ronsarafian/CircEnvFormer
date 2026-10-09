#!/usr/bin/env python3
"""Cluster precomputed event embeddings with k-means."""
from __future__ import annotations

import argparse
import numpy as np

from circenvformer import kmeans_cluster, normalized_wcss


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster event embeddings.")
    parser.add_argument("embeddings", help="Path to a .npy array of event embeddings.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="cluster_labels.npy")
    args = parser.parse_args()

    X = np.load(args.embeddings)
    labels, _ = kmeans_cluster(X, n_clusters=args.k, random_state=args.seed)
    np.save(args.output, labels)

    print(f"Embeddings: {X.shape}")
    print(f"Clusters: {args.k}")
    print(f"Normalized WCSS: {normalized_wcss(X, labels):.6f}")
    print(f"Saved labels: {args.output}")


if __name__ == "__main__":
    main()
