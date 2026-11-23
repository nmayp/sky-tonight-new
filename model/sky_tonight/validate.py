"""Compare the clear-sky model with an independent observation: SYNOP total cloud cover (oktas).

SYNOP's N includes high cloud, which METAR does not report. It is observed on only part of the hours
(manned/synoptic reports), so every number here says how many hours it rests on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import labels, model, train

CLEAR_N = 2  # at most 2 oktas of total cloud


def load_synop(path="data/synop_08181.csv") -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["hour"])
    s["hour"] = pd.DatetimeIndex(s.hour).tz_convert("UTC") if s.hour.dt.tz else pd.DatetimeIndex(s.hour).tz_localize("UTC")
    s = s.dropna(subset=["N"]).set_index("hour")
    s["clear_n"] = (s.N <= CLEAR_N).astype(int)
    return s[["N", "clear_n"]]


def main():
    d = train.dataset(target="metar")       # night hours with weather features + METAR label
    syn = load_synop()
    d = d.join(syn, how="inner")             # hours that have a total-cloud observation
    print(f"night hours with both METAR and SYNOP total cloud: {len(d)}")
    print(f"METAR 'clear' rate {d.clear.mean():.3f}; SYNOP 'clear' (N<=2) rate {d.clear_n.mean():.3f}")
    both = pd.crosstab(d.clear, d.clear_n, rownames=["METAR clear"], colnames=["SYNOP N<=2"])
    print(both.to_string())
    miss = ((d.clear == 1) & (d.clear_n == 0)).sum() / max(1, (d.clear == 1).sum())
    print(f"when METAR says clear, SYNOP finds N>=3 oktas in {miss:.1%} of hours (the high-cloud blind spot)")

    train_d, test_d = d[d.index.year < 2026], d[d.index.year == 2026]
    print(f"\nagainst SYNOP total cloud: train {len(train_d)} h (2023+2025), test {len(test_d)} h (2026)")
    # (a) the existing model, trained on METAR labels, scored against the SYNOP truth
    full = train.dataset(target="metar")
    metar_train = full[full.index.year < 2026]
    out = {}
    for name in ("raw forecast, calibrated", "logistic"):
        m = model.MODELS[name]().fit(metar_train, metar_train.clear.values)
        out[f"{name} (trained on METAR)"] = model.score(test_d.clear_n.values, m.predict(test_d))
    # (b) the same models retrained directly on the SYNOP label (small training set)
    for name in ("raw forecast, calibrated", "logistic"):
        m = model.MODELS[name]().fit(train_d, train_d.clear_n.values)
        out[f"{name} (trained on SYNOP)"] = model.score(test_d.clear_n.values, m.predict(test_d))
    out["climatology (SYNOP)"] = model.score(
        test_d.clear_n.values, model.Climatology().fit(train_d, train_d.clear_n.values).predict(test_d))
    print(pd.DataFrame(out).T[["brier", "logloss", "auc", "ece", "n"]].round(4).to_string())
    return out


if __name__ == "__main__":
    main()
