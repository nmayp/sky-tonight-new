"""Live forecasts for many places from the latest NOAA GFS run, for the multi-station model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import gfs

RES = "0p50"


def latest_cycle(now: datetime | None = None) -> datetime:
    """Most recent GFS cycle whose f003 file is already published (runs are ready about 4 hours after the cycle)."""
    now = now or datetime.now(timezone.utc)
    base = (now - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=(base.hour // 6) * 6)
    for k in range(0, 6):
        cyc = base - timedelta(hours=6 * k)
        url = gfs._url(cyc, 3, RES) + ".idx"
        try:
            if requests.head(url, headers=gfs.UA, timeout=30).status_code == 200:
                return cyc
        except requests.RequestException:
            pass
    raise RuntimeError("no recent GFS cycle found")


def site_features(df: pd.DataFrame, elev_m: float) -> pd.DataFrame:
    """Add the derived columns the multi-station model uses."""
    x = df.copy()
    x["dew_depression"] = x.temperature_2m - x.dew_point_2m
    idx = x.index
    x["month_sin"], x["month_cos"] = np.sin(2 * np.pi * idx.month / 12), np.cos(2 * np.pi * idx.month / 12)
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)
    x["elev_km"] = elev_m / 1000.0
    return x


def forecasts(sites: dict[str, dict], valid_hours: list[datetime], cycle: datetime | None = None,
              workers: int = 6) -> tuple[dict[str, pd.DataFrame], datetime]:
    """Model-ready features per place for the given valid times, all from one GFS cycle."""
    cycle = cycle or latest_cycle()
    points = {k: (s["lat"], s["lon"]) for k, s in sites.items()}
    # the 0.5 degree files exist every 3 hours: fetch the neighbouring steps and interpolate in time
    leads = set()
    for vt in valid_hours:
        h = (vt - cycle).total_seconds() / 3600
        leads |= {int(h // 3) * 3, int(-(-h // 3)) * 3}
    jobs = [(lead, cycle + timedelta(hours=lead)) for lead in sorted(leads) if 3 <= lead <= 120]

    def one(j):
        lead, vt = j
        return vt, gfs.points_forecast(cycle, lead, points, RES)

    rows = {k: {} for k in sites}
    with ThreadPoolExecutor(workers) as ex:
        for vt, res in ex.map(one, jobs):
            for k, v in res.items():
                rows[k][pd.Timestamp(vt).tz_convert("UTC") if pd.Timestamp(vt).tzinfo else pd.Timestamp(vt, tz="UTC")] = gfs.to_model_features(v)
    out = {}
    for k, r in rows.items():
        if r:
            df = pd.DataFrame.from_dict(r, orient="index").sort_index()
            want = pd.DatetimeIndex([pd.Timestamp(v).tz_convert("UTC") if pd.Timestamp(v).tzinfo else pd.Timestamp(v, tz="UTC") for v in valid_hours])
            df = df.reindex(df.index.union(want)).interpolate(method="time", limit_area="inside").reindex(want).dropna()
            out[k] = site_features(df, sites[k]["elev_m"])
    return out, cycle
