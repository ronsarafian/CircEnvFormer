"""ECMWF IFS benchmark preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm


def build_ifs_station_dataframe(root, ims_full_meta, target_times, nominal_lead=24):
    """Build 3-hour IFS precipitation at IMS station locations.

    For a nominal lead L, precipitation for a verification timestamp is
    derived from accumulated precipitation over the interval L->L+3h,
    matching the 3-hour IMS target.
    """
    root = Path(root)
    meta = ims_full_meta.copy()
    target_times = pd.DatetimeIndex(target_times).sort_values()
    rows = []

    n_total = len(target_times)
    n_processed = n_saved = n_missing_files = n_bad_time = 0
    print(f"\nBuilding IFS {nominal_lead}-hour benchmark")
    print(f"Requested timestamps: {n_total:,}")

    for valid_time in tqdm(target_times, desc=f"IFS {nominal_lead}h"):
        n_processed += 1

        # Select the most recent 00/12 UTC cycle whose age equals the nominal lead.
        if valid_time.hour < 12:
            init = valid_time.normalize()
        else:
            init = valid_time.normalize() + pd.Timedelta(hours=12)
        init -= pd.Timedelta(hours=nominal_lead)

        lead_start = int((valid_time - init) / pd.Timedelta(hours=1))
        if lead_start < nominal_lead:
            continue
        lead_start = nominal_lead + 3 * int(np.ceil((lead_start - nominal_lead) / 3))
        lead_end = lead_start + 3

        day_dir = root / f"{init.year:04d}" / f"{init.month:02d}" / f"{init.day:02d}"
        init_str = init.strftime("%Y-%m-%d_%H")
        f0 = day_dir / f"ifs_{init_str}_f{lead_start:02d}.pt"
        f1 = day_dir / f"ifs_{init_str}_f{lead_end:02d}.pt"

        if not f0.exists() or not f1.exists():
            n_missing_files += 1
            continue

        a = torch.load(f0, map_location="cpu", weights_only=False)
        b = torch.load(f1, map_location="cpu", weights_only=False)

        file_valid_time = pd.to_datetime(a["valid_time"], utc=True)
        if file_valid_time != valid_time:
            n_bad_time += 1
            continue

        tp_idx = a["channel_names"].index("tp:tp")
        tp = (b["features"][tp_idx] - a["features"][tp_idx]).clamp(min=0).numpy() * 1000.0

        lat = np.asarray(a["latitudes"])
        lon = np.asarray(a["longitudes"])
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            tp = tp[::-1]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            tp = tp[:, ::-1]

        interp = RegularGridInterpolator(
            (lat, lon),
            tp,
            bounds_error=False,
            fill_value=np.nan,
        )
        station_tp = interp(np.column_stack([meta["station_lat"].values, meta["station_lon"].values]))

        for station, value in zip(meta.index, station_tp):
            rows.append({
                "initialization_time": init,
                "valid_time": valid_time,
                "lead_start": lead_start,
                "lead_end": lead_end,
                "station": station,
                "tp_mm": float(value),
            })
        n_saved += 1

        if n_processed % 1000 == 0:
            print(
                f"Processed: {n_processed:,}/{n_total:,} | "
                f"Forecasts: {n_saved:,} | Missing: {n_missing_files:,} | "
                f"Bad time: {n_bad_time:,} | Rows: {len(rows):,}"
            )

    print("\n===============================")
    print(f"Nominal lead      : {nominal_lead} h")
    print(f"Requested         : {n_processed:,}")
    print(f"Forecasts created : {n_saved:,}")
    print(f"Missing forecasts : {n_missing_files:,}")
    print(f"Time mismatches   : {n_bad_time:,}")
    print(f"Total rows        : {len(rows):,}")
    print("===============================\n")

    return pd.DataFrame(rows)
