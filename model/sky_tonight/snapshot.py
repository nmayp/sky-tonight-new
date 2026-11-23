"""Forecast snapshot for the whole grid: latest NOAA GFS run, scored by the Spain-wide model, next ~10 days.

    python -m sky_tonight.snapshot --grid grid.json --model data/model_multi.pkl --out forecast.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd

from . import gfs, live_gfs

HOURS = (18, 21, 0, 3, 6)   # UT: evening to early morning in mainland Spain and the Canaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    grid = pd.DataFrame(json.load(open(a.grid)))
    pts = {r.wmo: (r.lat, r.lon) for r in grid.itertuples()}
    bundle = pickle.load(open(a.model, "rb"))
    cyc = live_gfs.latest_cycle()
    valid = []
    d0 = cyc.replace(hour=0)
    for dd in range(0, a.days + 1):
        for h in HOURS:
            t = d0 + timedelta(days=dd, hours=h)
            lead = int((t - cyc).total_seconds() // 3600)
            if 3 <= lead <= 240 and lead % 3 == 0:
                valid.append((lead, t))

    valid.sort(key=lambda x: x[1])

    def one(j):
        return j, gfs.points_forecast(cyc, j[0], pts, live_gfs.RES)

    out_p, out_t = [], []
    with ThreadPoolExecutor(6) as ex:
        for (lead, t), res in ex.map(one, valid):
            df = pd.DataFrame({k: gfs.to_model_features(v) for k, v in res.items()}).T
            df.index = pd.DatetimeIndex([t] * len(df))
            x = live_gfs.site_features(df, 0)
            x["elev_km"] = (grid.set_index("wmo").alt.reindex(res.keys()).values) / 1000.0
            p = pd.Series(bundle["model"].predict(x), index=list(res.keys())).reindex(grid.wmo)
            out_p.append([None if pd.isna(v) else int(round(v * 100)) for v in p.values])
            out_t.append({"utc": t.strftime("%Y-%m-%dT%H:00Z"), "lead_h": lead})
    json.dump({"cycle": cyc.strftime("%Y-%m-%dT%H:00Z"), "valid": out_t, "p": out_p}, open(a.out, "w"), separators=(",", ":"))
    print(len(out_t), "valid times for", len(grid), "points; cycle", cyc)


if __name__ == "__main__":
    from sky_tonight.snapshot import main as _m
    _m()
