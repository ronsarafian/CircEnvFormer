"""Clustering and cluster-compactness utilities."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def normalized_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute WCSS normalized by total sum of squares."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X must be 2-D and have one label per sample.")
    mean = np.mean(X, axis=0)
    tss = np.sum((X - mean) ** 2)
    if tss == 0:
        return 0.0

    wcss = 0.0
    for cluster in np.unique(labels):
        points = X[labels == cluster]
        centroid = points.mean(axis=0)
        wcss += np.sum((points - centroid) ** 2)
    return float(wcss / tss)


def kmeans_cluster(
    embeddings: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, KMeans]:
    """Cluster embeddings using the k-means procedure used in the analysis."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(np.asarray(embeddings))
    return labels, model
