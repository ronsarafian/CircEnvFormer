"""Rainfall forecast evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_daily_station(df_raw_3h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station-level 3-hour predictions to daily events."""
    def agg_gt(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float((x == 1).any())

    return (
        df_raw_3h.groupby(["day", "station", "region"], as_index=False)
        .agg(
            gt_event=("gt_event", agg_gt),
            pred_event=("pred_event", "max"),
            n_valid_3h=("gt_event", lambda x: x.notna().sum()),
            n_missing_3h=("gt_event", lambda x: x.isna().sum()),
            n_rain_3h=("gt_event", lambda x: np.nansum(x)),
            n_pred_rain_3h=("pred_event", "sum"),
        )
    )


def grouped_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute binary contingency statistics without casting missing values."""
    def metrics(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["gt_event", "pred_event"]).copy()
        gt = g["gt_event"].astype(int)
        pred = g["pred_event"].astype(int)

        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        tn = int(((gt == 0) & (pred == 0)).sum())

        pod = tp / (tp + fn) if (tp + fn) else np.nan
        far = fp / (tp + fp) if (tp + fp) else np.nan
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = pod
        f1 = (
            2 * precision * recall / (precision + recall)
            if pd.notna(precision) and pd.notna(recall) and (precision + recall)
            else np.nan
        )
        accuracy = (tp + tn) / len(g) if len(g) else np.nan
        bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan

        return pd.Series({
            "N": len(g), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "POD": pod, "FAR": far, "CSI": csi,
            "Precision": precision, "Recall": recall, "F1": f1,
            "Accuracy": accuracy, "Bias": bias,
        })

    return df.groupby(group_cols, dropna=False).apply(metrics).reset_index()


def compute_region_metrics(df_day: pd.DataFrame) -> pd.DataFrame:
    """Compute daily event metrics by region plus an all-dryland row."""
    region_day = (
        df_day.groupby(["day", "region"], as_index=False)
        .agg(gt_event=("gt_event", "max"), pred_event=("pred_event", "max"))
    )
    overall_day = (
        df_day.groupby("day", as_index=False)
        .agg(gt_event=("gt_event", "max"), pred_event=("pred_event", "max"))
    )
    overall_day["region"] = "overall"
    region_day = pd.concat([region_day, overall_day], ignore_index=True)
    return (
        grouped_metrics(region_day, ["region"])
        .sort_values("CSI", ascending=False)
        .reset_index(drop=True)
    )


def compute_region_metrics_3h(df_raw_3h: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-hour regional event metrics plus an all-dryland row."""
    def agg_gt(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float((x == 1).any())

    df = df_raw_3h.copy()
    if "region" not in df.columns:
        raise ValueError("df_raw_3h must contain a 'region' column.")
    df = df.dropna(subset=["region"])

    region_3h = (
        df.groupby(["valid_time", "region"], as_index=False)
        .agg(gt_event=("gt_event", agg_gt), pred_event=("pred_event", "max"))
    )
    overall_3h = (
        df.groupby("valid_time", as_index=False)
        .agg(gt_event=("gt_event", agg_gt), pred_event=("pred_event", "max"))
    )
    overall_3h["region"] = "overall"
    region_3h = pd.concat([region_3h, overall_3h], ignore_index=True)
    region_3h = region_3h.dropna(subset=["gt_event"])
    return (
        grouped_metrics(region_3h, ["region"])
        .sort_values("CSI", ascending=False)
        .reset_index(drop=True)
    )
