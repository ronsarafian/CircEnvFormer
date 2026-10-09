"""Rainfall event extraction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_rain_events(
    ims_full: pd.DataFrame,
    stations: list[str],
    threshold: float = 10.0,
    min_stations_above_threshold: int = 5,
    continue_threshold: float = 1.0,
    min_stations_continue: int = 2,
    delta_hours: int = 12,
    lead_time_hours: int = 0,
    window: int | None = None,
) -> list[pd.DatetimeIndex]:
    """Extract temporally continuous multi-station rainfall events.

    The defaults mirror the exploratory notebook event detector. The start
    criterion and continuation criterion can be adjusted for other studies.
    """
    timestamps = pd.DatetimeIndex(ims_full.index)
    shifted = timestamps + pd.Timedelta(hours=lead_time_hours)
    data = ims_full.loc[shifted, stations].copy()
    data.index = timestamps

    if window is not None and window > 1:
        data = data.rolling(window=window, center=True, min_periods=1).mean()

    stations_above_start = (data > threshold).sum(axis=1)
    stations_above_continue = (data > continue_threshold).sum(axis=1)

    events: list[pd.DatetimeIndex] = []
    current_event: list[pd.Timestamp] = []
    in_event = False
    last_wet_time: pd.Timestamp | None = None
    dry_gap = pd.Timedelta(hours=delta_hours)

    for ts in timestamps:
        if not in_event:
            if stations_above_start.loc[ts] >= min_stations_above_threshold:
                current_event = [ts]
                in_event = True
                last_wet_time = ts
            continue

        current_event.append(ts)
        if stations_above_continue.loc[ts] >= min_stations_continue:
            last_wet_time = ts

        if last_wet_time is not None and ts - last_wet_time > dry_gap:
            event_times = [t for t in current_event if t <= last_wet_time]
            if event_times:
                events.append(pd.DatetimeIndex(event_times))
            current_event = []
            in_event = False
            last_wet_time = None

    if in_event and last_wet_time is not None:
        event_times = [t for t in current_event if t <= last_wet_time]
        if event_times:
            events.append(pd.DatetimeIndex(event_times))

    return events
