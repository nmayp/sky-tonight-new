"""Train and test the clear-sky model on many Spanish stations, with regions held out.

Features: GFS forecast issued at 12 UT for the evening/night ahead (real forecast, leads 9-15 h).
Target: SYNOP total cloud <= 2 oktas at the station.
Run from the repository root with the CSVs written by the download/feature jobs:
    python -m sky_tonight.train_multi --synop 'synop_all_*.csv' --feats feats.csv --stations stations.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .model import ece
from .multi import FEATS

WEATHER = FEATS
EXTRA = ["dew_depression", "month_sin", "month_cos", "hour_sin", "hour_cos"]
SITEF = ["elev_km", "lat", "lon"]


def load(synop_glob: str, feats_csv: str, stations_json: str) -> pd.DataFrame:
    s = pd.concat([pd.read_csv(f, dtype={"wmo": str}) for f in glob.glob(synop_glob)]).drop_duplicates(["wmo", "utc"])
    s["wmo"] = s.wmo.astype(str).str.zfill(5)
    f = pd.read_csv(feats_csv, dtype={"wmo": str}).drop_duplicates(["wmo", "valid_utc"])
    f["wmo"] = f.wmo.str.zfill(5)
    d = f.merge(s, left_on=["wmo", "valid_utc"], right_on=["wmo", "utc"])
    d["t"] = pd.to_datetime(d.valid_utc, utc=True)
    st = {x["wmo"]: x for x in json.load(open(stations_json))}
    d["lat"] = d.wmo.map(lambda w: st[w]["lat"])
    d["lon"] = d.wmo.map(lambda w: st[w]["lon"])
    d["elev_km"] = d.wmo.map(lambda w: (st[w]["alt"] or 0) / 1000.0)
    alt = pd.Series(0.0, index=d.index)
    for w, g in d.groupby("wmo"):
        loc = EarthLocation(lat=st[w]["lat"] * u.deg, lon=st[w]["lon"] * u.deg, height=(st[w]["alt"] or 0) * u.m)
        tt = Time(g.t.dt.tz_convert("UTC").dt.tz_localize(None).values)
        alt[g.index] = get_sun(tt).transform_to(AltAz(obstime=tt, location=loc)).alt.deg
    d["sun_alt"] = alt
    d = d[d.sun_alt < -6].copy()
    d["clear"] = (d.N <= 2).astype(int)
    d["dew_depression"] = d.temperature_2m - d.dew_point_2m
    m, h = d.t.dt.month, d.t.dt.hour
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return d.reset_index(drop=True)


class Raw:
    def fit(self, X, y):
        self.iso = IsotonicRegression(y_min=0, y_max=1, increasing=False, out_of_bounds="clip").fit(X["cloud_cover"], y)
        return self

    def predict(self, X):
        return self.iso.predict(X["cloud_cover"])


class Naive:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(X["cloud_cover"].values < 25, 0.9, 0.1)


class Boost:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y, calib_frac=0.25):
        order = np.argsort(X["t"].values)       # calibrate on the latest part of the training data
        X, y = X.iloc[order], np.asarray(y)[order]
        cut = int(len(X) * (1 - calib_frac))
        self.m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=60,
                                                l2_regularization=1.0, random_state=0)
        self.m.fit(X[self.cols].iloc[:cut], y[:cut])
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            self.m.predict_proba(X[self.cols].iloc[cut:])[:, 1], y[cut:])
        return self

    def predict(self, X):
        return self.iso.predict(self.m.predict_proba(X[self.cols])[:, 1])


def score(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p), "auc": roc_auc_score(y, p),
            "ece": ece(np.asarray(y), p), "n": int(len(y)), "base_rate": float(np.mean(y))}


def models():
    w = WEATHER + EXTRA
    return {"naive": Naive(), "raw calibrated": Raw(), "boosted (weather)": Boost(w),
            "boosted (+elevation)": Boost(w + ["elev_km"]), "boosted (+elev, lat, lon)": Boost(w + SITEF)}


def region(d: pd.DataFrame) -> pd.Series:
    r = pd.Series("interior/other", index=d.index)
    r[(d.lat < 38.5) & (d.lon > -7.5) & (d.lon < -1)] = "south"
    r[(d.lat > 42.3) & (d.lon < -1.0)] = "north"
    r[(d.lat > 40.4) & (d.lon > 0.15) & (d.lon < 3.4)] = "catalonia"
    r[d.wmo.str.startswith("60")] = "canary"
    return r


def evaluate(d: pd.DataFrame) -> dict:
    out = {}
    tr, te = d[d.t.dt.year < 2026], d[d.t.dt.year == 2026]
    out["time_split"] = {"train_rows": len(tr), "test_rows": len(te), "stations": int(d.wmo.nunique()), "scores": {}}
    for name, m in models().items():
        m.fit(tr, tr.clear.values)
        out["time_split"]["scores"][name] = score(te.clear.values, m.predict(te))
    d = d.assign(region=region(d))
    out["region_holdout"] = {}
    for reg in ["catalonia", "south", "north", "canary"]:
        trr = d[(d.t.dt.year < 2026) & (d.region != reg)]
        ter = d[(d.t.dt.year == 2026) & (d.region == reg)]
        if len(ter) < 150 or len(trr) < 500:
            continue
        res = {"test_rows": len(ter), "test_stations": int(ter.wmo.nunique())}
        for name, m in models().items():
            m.fit(trr, trr.clear.values)
            res[name] = score(ter.clear.values, m.predict(ter))
        trin = d[d.t.dt.year < 2026]
        mm = models()["boosted (weather)"].fit(trin, trin.clear.values)
        res["boosted (weather) [region seen in training]"] = score(ter.clear.values, mm.predict(ter))
        out["region_holdout"][reg] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synop", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out", default="data/multi_scores.json")
    ap.add_argument("--fit-final", action="store_true")
    a = ap.parse_args()
    d = load(a.synop, a.feats, a.stations)
    print(len(d), "rows,", d.wmo.nunique(), "stations,", d.t.min().date(), "to", d.t.max().date(),
          "clear rate", round(d.clear.mean(), 3))
    res = evaluate(d)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    ts = res["time_split"]
    print("\nTIME SPLIT (test = 2026, all stations)", ts["train_rows"], "train /", ts["test_rows"], "test")
    print(pd.DataFrame(ts["scores"]).T[["brier", "auc", "ece"]].round(4).to_string())
    for reg, r in res["region_holdout"].items():
        print(f"\nREGION HELD OUT: {reg} ({r['test_stations']} stations, {r['test_rows']} test rows)")
        print(pd.DataFrame({k: v for k, v in r.items() if isinstance(v, dict)}).T[["brier", "auc", "ece"]].round(4).to_string())
    if a.fit_final:
        cols = WEATHER + EXTRA + ["elev_km"]
        m = Boost(cols).fit(d, d.clear.values)
        raw = Raw().fit(d, d.clear.values)
        pickle.dump({"name": "boosted (+elevation), multi-station", "model": m, "raw": raw, "cols": cols,
                     "stations": int(d.wmo.nunique())}, open("data/model_multi.pkl", "wb"))
        print("saved data/model_multi.pkl")


if __name__ == "__main__":
    # re-import so pickled classes belong to sky_tonight.train_multi, not __main__
    from sky_tonight.train_multi import main as _main
    _main()
