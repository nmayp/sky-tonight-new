"""Train on 2023+2025 nights, test on 2026 nights (never seen). Prints a scoreboard."""
from __future__ import annotations

import json
import pickle
import sys

import pandas as pd

from . import features, labels, model


def dataset(weather_csv="data/openmeteo_archive.csv", metar_csv="data/metar_lebl.csv",
            target: str = "synop"):
    """Night hours with forecast features. target='synop': total cloud <= 2 oktas (includes high cloud,
    observed on part of the hours); target='metar': no low/middle cloud (every hour, misses high cloud)."""
    w = pd.read_csv(weather_csv, parse_dates=["time"]).set_index("time")
    w.index = pd.DatetimeIndex(w.index).tz_localize("UTC")
    y = labels.load_metar(metar_csv)
    y.index = pd.DatetimeIndex(y.index).tz_localize("UTC")
    X = features.build(w)
    d = X.join(y, how="inner")
    if target == "synop":
        from .validate import load_synop
        d = d.join(load_synop(), how="inner")
        d["clear"] = d.clear_n          # the target used everywhere downstream
    d = d[d.sun_alt < -6]  # astronomical-ish night only: when stars could be seen
    return d.dropna(subset=features.FEATURES + ["clear"])


def main():
    d = dataset()
    train, test = d[d.index.year < 2026], d[d.index.year == 2026]
    print(f"night hours: train {len(train)} (2023+2025), test {len(test)} (2026)")
    print(f"clear rate: train {train.clear.mean():.3f}, test {test.clear.mean():.3f}")
    rows, fitted = {}, {}
    for name, cls in model.MODELS.items():
        m = cls().fit(train, train.clear.values)
        p = m.predict(test)
        rows[name] = model.score(test.clear.values, p)
        fitted[name] = m
    out = pd.DataFrame(rows).T[["brier", "logloss", "auc", "ece", "n"]]
    print(out.round(4).to_string())
    best = out.brier.idxmin()
    print("best by Brier:", best)
    print(model.reliability(test.clear.values, fitted[best].predict(test)).round(3).to_string(index=False))
    # bootstrap CI for Brier difference vs the raw-forecast baseline, by whole nights
    import numpy as np
    rng = np.random.default_rng(0)
    nights = test.index.floor("D") - pd.Timedelta(hours=0)
    day_ids = pd.factorize(nights)[0]
    pb = fitted[best].predict(test); pr = fitted["raw forecast, calibrated"].predict(test)
    y = test.clear.values
    eb, er = (pb - y) ** 2, (pr - y) ** 2
    nd = day_ids.max() + 1
    diffs = []
    for _ in range(2000):
        pick = rng.integers(0, nd, nd)
        w = np.bincount(pick, minlength=nd)[day_ids]
        diffs.append((w * (er - eb)).sum() / w.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"Brier gain over calibrated raw forecast: {np.mean(er - eb):+.4f} (95% CI {lo:+.4f}..{hi:+.4f}, resampled by night)")
    json.dump({"scores": rows, "best": best, "gain_vs_raw": float(np.mean(er - eb)),
               "gain_ci": [float(lo), float(hi)],
               "train_years": [2023, 2025], "test_year": 2026},
              open("data/scores.json", "w"), indent=2, default=float)
    refit = model.MODELS[best]().fit(d, d.clear.values)
    raw = model.RawCloudBaseline().fit(d, d.clear.values)
    pickle.dump({"name": best, "model": refit, "raw": raw}, open("data/model.pkl", "wb"))


if __name__ == "__main__":
    main()
