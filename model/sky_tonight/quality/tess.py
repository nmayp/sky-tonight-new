"""Night-sky brightness from the Stars4All TESS-W photometer network (Spain).

The network's public dashboard (https://tess.dashboards.stars4all.eu) is open and its database can be queried
without a key. Photometers are named 'starsN'; their coordinates are in the dashboard's data-source list (one entry
per place the photometer has stood; photometers that moved more than 20 km are dropped). Hourly means of the sky
brightness (mag/arcsec2, larger = darker) are fetched one photometer at a time with a pause.
Credit: Stars4All / TESS-W network (https://tess.stars4all.eu), data openly published on its dashboard.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://tess.dashboards.stars4all.eu"
UA = {"User-Agent": "sky-tonight-research/0.3 (non-commercial research project; low volume)"}


def km(lat1, lon1, lat2, lon2) -> float:
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def spanish_photometers(settings: dict) -> dict[str, tuple[float, float]]:
    """name -> (lat, lon) for photometers labelled Spain whose entries all lie within 20 km of each other."""
    pos: dict[str, list[tuple[float, float]]] = {}
    for k, v in settings["datasources"].items():
        if "(Spain)" not in k or not k.startswith("stars"):
            continue
        p = (v.get("jsonData") or {}).get("position")
        if p and p.get("latitude") is not None:
            pos.setdefault(k.split(" ")[0], []).append((float(p["latitude"]), float(p["longitude"])))
    out = {}
    for n, ps in pos.items():
        if all(km(*ps[0], *q) < 20 for q in ps):
            out[n] = ps[-1]
    return out


def _query(q: str, retries: int = 3) -> dict:
    for k in range(retries):
        try:
            r = requests.get(BASE + "/api/datasources/proxy/2/query", params={"db": "telegraf", "q": q}, headers=UA, timeout=170)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError("TESS dashboard database did not answer")


def hourly(name: str, start: str, end: str) -> pd.Series:
    q = (f"SELECT mean(\"mag\") FROM \"mqtt_consumer\" WHERE \"name\"='{name}' AND time >= '{start}T00:00:00Z' "
         f"AND time < '{end}T00:00:00Z' GROUP BY time(1h) fill(none)")
    s = _query(q)["results"][0].get("series")
    if not s:
        return pd.Series(dtype=float)
    v = s[0]["values"]
    return pd.Series([x[1] for x in v], index=pd.DatetimeIndex([x[0] for x in v]).tz_convert("UTC") if pd.DatetimeIndex([v[0][0]]).tz else pd.DatetimeIndex([x[0] for x in v], tz="UTC"), name=name)


def fetch_all(names: list[str], out_path: str, start: str = "2023-01-01", end: str = "2026-11-02", pause: float = 2.0) -> None:
    import pickle
    from pathlib import Path
    have = pickle.load(open(out_path, "rb")) if Path(out_path).exists() else {}
    for i, n in enumerate(names):
        if n in have:
            continue
        try:
            have[n] = hourly(n, start, end)
        except RuntimeError as exc:
            print("skip", n, exc, flush=True)
            continue
        print(i, n, len(have[n]), flush=True)
        pickle.dump(have, open(out_path, "wb"))
        time.sleep(pause)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="tess_counts.json from the probe (monthly reading counts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions-out", required=True)
    ap.add_argument("--min-months", type=int, default=12)
    a = ap.parse_args()
    st = requests.get(BASE + "/api/frontend/settings", headers=UA, timeout=60).json()
    pos = spanish_photometers(st)
    cnt = json.load(open(a.counts))["counts"]
    names = sorted(n for n in pos if n in cnt and sum(1 for v in cnt[n].values() if v >= 1000) >= a.min_months)
    json.dump({n: pos[n] for n in names}, open(a.positions_out, "w"))
    print(len(names), "photometers")
    fetch_all(names, a.out)
