"""Build the v2 training/test table: as-issued GFS fields + aerosol state + night-time SYNOP labels.

    python -m sky_tonight.quality.build_table --synop 'work/synop_*.csv' --gfs work/gfs_table.csv \
        --stations data/quality/stations_spain_synop.json --out work/table.csv.gz

Steps (all documented in docs/QUALITY_V2.md): join the GFS table to SYNOP on (station, valid hour); keep rows with the
Sun more than 6 degrees below the horizon; pick stations with enough labelled nights; attach the aerosol state.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aq, features as F
from .synop_full import poor_transparency

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"


def load_synop(pattern: str) -> pd.DataFrame:
    s = pd.concat([pd.read_csv(f, dtype={"wmo": str}) for f in sorted(glob.glob(pattern))], ignore_index=True)
    s["wmo"] = s.wmo.astype(str).str.zfill(5)
    s = s.drop_duplicates(["wmo", "utc"]).copy()
    s["haze"] = [poor_transparency(None if pd.isna(v) else v, None if pd.isna(w) else w, int(o))
                 for v, w, o in zip(s.vv_km, s.ww, s.wx_obs)]
    return s


def select_stations(s: pd.DataFrame, min_train: int = 250, min_test: int = 80) -> list[str]:
    """Stations with enough labelled night rows (cloud reported) before and after the split."""
    t = pd.to_datetime(s.utc, utc=True)
    lab = s[s.N.notna()]
    tl = pd.to_datetime(lab.utc, utc=True)
    a = lab[tl < pd.Timestamp(TEST_START, tz="UTC")].groupby("wmo").size()
    b = lab[tl >= pd.Timestamp(TEST_START, tz="UTC")].groupby("wmo").size()
    ok = a[a >= min_train].index.intersection(b[b >= min_test].index)
    return sorted(ok)


def load_gfs(path: str, stations: list[str]) -> pd.DataFrame:
    """`path` may list several CSVs separated by commas (the main pass and the dusk/dawn pass)."""
    parts = []
    for one in path.split(","):
        for ch in pd.read_csv(one, dtype={"wmo": str}, chunksize=500_000):
            ch["wmo"] = ch.wmo.str.zfill(5)
            parts.append(ch[ch.wmo.isin(stations)])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["wmo", "valid_utc", "cycle"])


def attach_aerosol(d: pd.DataFrame, hist: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """dust_0/aod_0/pm10_0: state at 00 UT of the cycle's day (12 h before the 12 UT cycle).
    dust_v/aod_v/pm10_v: best-estimate at the valid hour (hindsight; only for the ablation)."""
    d = d.copy()
    for c in ("dust", "aod", "pm10"):
        d[f"{c}_0"] = np.nan
        d[f"{c}_v"] = np.nan
    cyc = pd.to_datetime(d.cycle, utc=True)
    base = cyc.dt.normalize()
    for w, idx in d.groupby("wmo").groups.items():
        h = hist.get(w)
        if h is None:
            continue
        for src, name in (("dust", "dust"), ("aerosol_optical_depth", "aod"), ("pm10", "pm10")):
            ser = h[src]
            d.loc[idx, f"{name}_0"] = ser.reindex(pd.DatetimeIndex(base.loc[idx])).values
            d.loc[idx, f"{name}_v"] = ser.reindex(pd.DatetimeIndex(d.t.loc[idx])).values
    return d


def build(synop_glob: str, gfs_csv: str, stations_json: str, aq_cache: str | None = None, stations: list[str] | None = None) -> pd.DataFrame:
    st = {x["wmo"]: x for x in json.load(open(stations_json))}
    s = load_synop(synop_glob)
    keep = [w for w in (stations or select_stations(s)) if w in st]       # need coordinates for the GFS point
    print(len(keep), "stations kept of", s.wmo.nunique(), "with synop rows")
    g = load_gfs(gfs_csv, keep)
    d = g.merge(s[["wmo", "utc", "N", "vv_km", "ww", "wx_obs", "haze"]], left_on=["wmo", "valid_utc"], right_on=["wmo", "utc"], how="inner")
    d["t"] = pd.to_datetime(d.valid_utc, utc=True)
    d["cyc_t"] = pd.to_datetime(d.cycle, utc=True)
    d["lat"] = d.wmo.map(lambda w: st[w]["lat"])
    d["lon"] = d.wmo.map(lambda w: st[w]["lon"])
    d["elev_km"] = d.wmo.map(lambda w: (st[w]["alt"] or 0) / 1000.0)
    d["sun_alt"] = F.sun_altitude(pd.DatetimeIndex(d.t), d.lat.values, d.lon.values)
    d = d[d.sun_alt < -6].copy()
    d = F.add_time_terms(d)
    d["clear"] = np.where(d.N.notna(), (d.N <= 2).astype(float), np.nan)
    d["oktas"] = d.N
    d["night"] = (d.t - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")          # night id: date of the evening before dawn
    d["same_night"] = d.lead_h.isin([6, 9, 12, 15, 18])
    # aerosol state
    hist = pd.read_pickle(aq_cache) if aq_cache and Path(aq_cache).exists() else {}
    missing = [w for w in keep if w not in hist]
    if missing:
        hist.update(aq.history({w: (st[w]["lat"], st[w]["lon"]) for w in missing}, "2022-12-01", "2026-11-05"))
        if aq_cache:
            pd.to_pickle(hist, aq_cache)
    d = attach_aerosol(d, hist)
    d["region"] = F.region(d)
    return d.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synop", required=True)
    ap.add_argument("--gfs", required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--aq-cache", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build(a.synop, a.gfs, a.stations, a.aq_cache)
    d.to_csv(a.out, index=False)
    print(len(d), "rows,", d.wmo.nunique(), "stations,", d.t.min(), "to", d.t.max())
    print("clear rate", round(d.clear.mean(), 3), "haze rate", round(d.haze.mean(), 3), "mean oktas", round(d.oktas.mean(), 2))


if __name__ == "__main__":
    main()
