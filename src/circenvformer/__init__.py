"""Core utilities for CircEnvFormer."""

from .model import (
    PatchEmbedding,
    MonthEmbedding,
    MultiheadAttentionWithReturn,
    TransformerEncoderLayerWithAttn,
    CircEnvEncoder,
    RainfallClassifier,
)
from .events import extract_rain_events
from .evaluation import (
    aggregate_daily_station,
    grouped_metrics,
    compute_region_metrics,
    compute_region_metrics_3h,
)
from .clustering import normalized_wcss, kmeans_cluster
from .ifs import build_ifs_station_dataframe

__all__ = [
    "PatchEmbedding",
    "MonthEmbedding",
    "MultiheadAttentionWithReturn",
    "TransformerEncoderLayerWithAttn",
    "CircEnvEncoder",
    "RainfallClassifier",
    "extract_rain_events",
    "aggregate_daily_station",
    "grouped_metrics",
    "compute_region_metrics",
    "compute_region_metrics_3h",
    "normalized_wcss",
    "kmeans_cluster",
    "build_ifs_station_dataframe",
]
