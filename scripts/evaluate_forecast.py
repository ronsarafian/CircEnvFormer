#!/usr/bin/env python3
"""Evaluate station-level rainfall forecasts from a prepared CSV file."""
from __future__ import annotations

import argparse
import pandas as pd

from circenvformer import compute_region_metrics, compute_region_metrics_3h


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CircEnvFormer rainfall forecasts.")
    parser.add_argument("csv", help="CSV containing valid_time/day, region, gt_event and pred_event.")
    parser.add_argument("--three-hour", action="store_true")
    parser.add_argument("--output", default="forecast_metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["valid_time"] if "valid_time" in pd.read_csv(args.csv, nrows=0).columns else None)
    metrics = compute_region_metrics_3h(df) if args.three_hour else compute_region_metrics(df)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
