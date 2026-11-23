"""Dust and aerosol from the Open-Meteo air-quality API (CAMS model fields).

Open-Meteo: https://open-meteo.com (free for non-commercial use, attribution required; data: Copernicus Atmosphere
Monitoring Service, CAMS). Variables: dust (ug/m3), aerosol_optical_depth (550 nm), pm10 (ug/m3).

What the history is: for past dates the API serves a best-estimate series stitched from the CAMS model runs (the
analysis-like first hours of each run), not the forecast that was available a day earlier. Using the value at the
forecast's valid time would therefore be hindsight (it flatters the dust input). The training table here uses the
value 12 h BEFORE the forecast cycle (00 UT of the cycle's day) as the dust state known when the forecast is issued;
the value at the valid time is built too, only to measure how much hindsight would flatter the result.
Aerosol optical depth history starts in late 2022; dust and pm10 earlier.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
UA = {"User-Agent": "sky-tonight-research/0.3 (non-commercial research project; low volume)"}
VARS = ["dust", "aerosol_optical_depth", "pm10"]


def parse_response(js) -> list[pd.DataFrame]:
    """One DataFrame (UTC hourly index, columns VARS) per location, in request order."""
    items = js if isinstance(js, list) else [js]
    out = []
    for it in items:
        h = it["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(h["time"]), tz="UTC")
        out.append(pd.DataFrame({v: np.array(h[v], dtype=float) for v in VARS if v in h}, index=idx))
    return out


def _get(params: dict, retries: int = 4):
    for k in range(retries):
        try:
            r = requests.get(URL, params=params, headers=UA, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(30 * (k + 1))
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    raise RuntimeError("air-quality API did not answer")


def history(points: dict[str, tuple[float, float]], start: str, end: str, batch: int = 10, pause: float = 1.0) -> dict[str, pd.DataFrame]:
    keys = list(points)
    out = {}
    for i in range(0, len(keys), batch):
        ks = keys[i:i + batch]
        js = _get({"latitude": ",".join(f"{points[k][0]:.4f}" for k in ks), "longitude": ",".join(f"{points[k][1]:.4f}" for k in ks),
                   "hourly": ",".join(VARS), "start_date": start, "end_date": end, "timezone": "GMT"})
        for k, df in zip(ks, parse_response(js)):
            out[k] = df
        time.sleep(pause)
    return out


def forecast(points: dict[str, tuple[float, float]], past_days: int = 2, forecast_days: int = 7, batch: int = 100,
             pause: float = 1.0) -> dict[str, pd.DataFrame]:
    """Recent past (model analysis-like) plus the forecast, for the live snapshot."""
    keys = list(points)
    out = {}
    for i in range(0, len(keys), batch):
        ks = keys[i:i + batch]
        js = _get({"latitude": ",".join(f"{points[k][0]:.4f}" for k in ks), "longitude": ",".join(f"{points[k][1]:.4f}" for k in ks),
                   "hourly": ",".join(VARS), "past_days": past_days, "forecast_days": forecast_days, "timezone": "GMT"})
        for k, df in zip(ks, parse_response(js)):
            out[k] = df
        time.sleep(pause)
    return out


def at(df: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Values at the given UTC hours (NaN where missing)."""
    return df.reindex(times)
