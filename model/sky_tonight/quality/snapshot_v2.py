"""One command that rebuilds forecast_v2.json: latest 12 UT GFS run (NOAA open data) + CAMS aerosol (Open-Meteo)
scored by the saved v2 model for the grid of grid.json, next ~10 days.

    python -m sky_tonight.quality.snapshot_v2 --grid ../portfolio/public/sky/grid.json \
        --model data/quality_v2.pkl --out ../portfolio/public/sky/forecast_v2.json

Why NOAA GFS and not the Open-Meteo weather API for the weather inputs: the model was trained on as-issued NOAA GFS
(12 UT cycle, 0.5 degree), so scoring the same fields keeps training and live inputs identical; Open-Meteo's
weather endpoints are also capped at 10 000 calls a day per address. Open-Meteo is used for the air-quality
(dust, aerosol optical depth, pm10) fields. Output format is the one of forecast.json plus c, h, dust and aod.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import aq, features as F, gfs_table as G

HOURS = (18, 21, 0, 3, 6)          # UT, evening to early morning in mainland Spain and the Canaries
CYCLE_HOUR = 12                    # the model was trained on 12 UT cycles


def latest_cycle(now: datetime | None = None, hour: int = CYCLE_HOUR, last_lead: int = 240) -> datetime:
    """Most recent cycle at `hour` UT whose last needed file is already published."""
    now = now or datetime.now(timezone.utc)
    day = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if day > now:
        day -= timedelta(days=1)
    for k in range(4):
        cyc = day - timedelta(days=k)
        r = requests.head(G.url_for(cyc, last_lead) + ".idx", headers=G.UA, timeout=30)
        if r.status_code == 200:
            return cyc
    raise RuntimeError("no recent GFS cycle found")


def valid_times(cycle: datetime, days: int = 10) -> list[tuple[int, datetime]]:
    out = []
    d0 = cycle.replace(hour=0)
    for dd in range(0, days + 1):
        for h in HOURS:
            t = d0 + timedelta(days=dd, hours=h)
            lead = int((t - cycle).total_seconds() // 3600)
            if 3 <= lead <= 240 and lead % 3 == 0:
                out.append((lead, t))
    return sorted(out, key=lambda x: x[1])


def feature_frame(fields: dict[str, np.ndarray], grid: pd.DataFrame, t: datetime, lead: int, aero0: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(G.derive(fields))
    d["lat"], d["lon"] = grid.lat.values, grid.lon.values
    d["elev_km"] = grid.alt.values / 1000.0
    d["lead_h"] = lead
    d["t"] = pd.Timestamp(t)
    d = F.add_time_terms(d)
    for c in aero0.columns:
        d[c] = aero0[c].values
    return d


def aerosol_state(aqd: dict[str, pd.DataFrame], keys: list[str], when: pd.Timestamp) -> pd.DataFrame:
    """dust_0, aod_0, pm10_0 at `when` (12 h before the cycle), NaN where the API gave nothing."""
    rows = []
    for k in keys:
        h = aqd.get(k)
        r = h.reindex([when]).iloc[0] if h is not None else pd.Series(dtype=float)
        rows.append([r.get("dust", np.nan), r.get("aerosol_optical_depth", np.nan), r.get("pm10", np.nan)])
    return pd.DataFrame(rows, columns=["dust_0", "aod_0", "pm10_0"])


def build(grid_path: str, model_path: str, out_path: str, days: int = 10, workers: int = 6, cycle: datetime | None = None) -> dict:
    g = json.load(open(grid_path))
    grid = pd.DataFrame({"lat": g["lat"], "lon": g["lon"], "alt": g["alt"]})
    keys = [str(i) for i in range(len(grid))]
    pts = G.PointSet({k: (la, lo) for k, la, lo in zip(keys, grid.lat, grid.lon)})
    bundle = pickle.load(open(model_path, "rb"))
    cyc = cycle or latest_cycle()
    vt = valid_times(cyc, days)
    try:
        aqd = aq.forecast({k: (la, lo) for k, la, lo in zip(keys, grid.lat, grid.lon)}, past_days=2, forecast_days=7)
        aerosol_ok = True
    except Exception as exc:                       # the model reads missing aerosol as unknown
        print("air-quality API failed, aerosol inputs left empty:", repr(exc)[:100], file=sys.stderr)
        aqd, aerosol_ok = {}, False
    aero0 = aerosol_state(aqd, keys, pd.Timestamp(cyc - timedelta(hours=12)))

    def one(j):
        lead, t = j
        return j, G.raw_fields(cyc, lead, pts)

    P, C, H, DU, AO, valid = [], [], [], [], [], []
    with ThreadPoolExecutor(workers) as ex:
        for (lead, t), raw in ex.map(one, vt):
            x = feature_frame(raw, grid, t, lead, aero0)
            p = bundle["clear"].predict(x)
            c = bundle["oktas"].predict(x)
            h = bundle["haze"].predict(x)
            P.append([int(round(v * 100)) for v in p])
            C.append([int(round(v * 10)) for v in c])
            H.append([int(round(v * 100)) for v in h])
            av = aerosol_state(aqd, keys, pd.Timestamp(t))
            DU.append([None if np.isnan(v) else int(round(v)) for v in av.dust_0])
            AO.append([None if np.isnan(v) else int(round(v * 100)) for v in av.aod_0])
            valid.append({"utc": t.strftime("%Y-%m-%dT%H:00Z"), "lead_h": lead})
    out = {"cycle": cyc.strftime("%Y-%m-%dT%H:00Z"), "valid": valid, "p": P, "c": C, "h": H, "dust": DU, "aod": AO,
           "units": {"p": "% clear", "c": "oktas x10", "h": "% poor transparency", "dust": "ug/m3 (CAMS, at the valid hour)", "aod": "x100 (550 nm)"},
           "model": bundle.get("version"), "aerosol_inputs": "ok" if aerosol_ok else "missing"}
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print(len(valid), "valid times for", len(grid), "cells; cycle", cyc, "aerosol", out["aerosol_inputs"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--model", default="data/quality_v2.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    build(a.grid, a.model, a.out, a.days)


if __name__ == "__main__":
    main()
