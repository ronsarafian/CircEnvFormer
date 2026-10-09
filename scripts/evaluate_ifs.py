#!/usr/bin/env python3
"""Build station-level IFS precipitation forecasts for evaluation."""
from __future__ import annotations

import argparse
import pandas as pd

from circenvformer import build_ifs_station_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IFS station-level forecast table.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata", required=True, help="CSV with station_lat and station_lon indexed by station.")
    parser.add_argument("--times", required=True, help="CSV containing a valid_time column.")
    parser.add_argument("--lead", type=int, choices=[24, 48], default=24)
    parser.add_argument("--output", default="ifs_station_forecasts.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, index_col=0)
    times = pd.read_csv(args.times, parse_dates=["valid_time"])["valid_time"]
    result = build_ifs_station_dataframe(args.root, meta, times, nominal_lead=args.lead)
    result.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
