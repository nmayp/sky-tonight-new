"""Train and test the v2 sky-quality models (clear, cloud oktas, poor transparency) and write the scores.

    python -m sky_tonight.quality.train_quality --table work/table.csv.gz --out data/quality_v2_scores.json \
        --model-out data/quality_v2.pkl [--public-dir ../portfolio/public/sky]

Protocol: train 2023-01 .. 2025-12, test 2026-01 .. 2026-10, nights only (Sun below -6 degrees). Hyper-parameters are
chosen among four settings on 2025 with models trained on 2023-2024; the 2026 test period is never used to choose
anything. Intervals resample whole nights.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import evaluate as E, features as F
from .models import PARAM_GRID, Boost, BoostReg, Clim, RawIso, RawOktas

TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-11-01", tz="UTC")
VAL_START = pd.Timestamp("2025-01-01", tz="UTC")
SHORT = (6, 9, 12, 15, 18)          # same-night leads used for training the headline models
CORE = (9, 12, 15)                  # leads v1 was tested on
TARGETS = {"clear": "clear", "haze": "haze", "oktas": "oktas"}
V1 = "v1-equivalent"
V2 = "v2"
RAW = "plain forecast (calibrated)"
CLIM = "climatology (station x month)"
CLIM_M = "climatology (month, all stations)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_table(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False, dtype={"wmo": str})
    d["wmo"] = d.wmo.str.zfill(5)
    d["t"] = pd.to_datetime(d.t, utc=True)
    return d


def rows_for(d: pd.DataFrame, target: str) -> pd.DataFrame:
    return d[d[TARGETS[target]].notna()]


def is_cls(target: str) -> bool:
    return target in ("clear", "haze")


def make(target: str, cols, params):
    return Boost(cols, params) if is_cls(target) else BoostReg(cols, params)


def raw_model(target: str):
    if target == "clear":
        return RawIso("cloud_cover", increasing=False)
    if target == "haze":
        return RawIso("vis_m", increasing=False)
    return RawIso("cloud_cover", increasing=True, lo=0.0, hi=8.0)


def loss_name(target: str) -> str:
    return "logloss" if is_cls(target) else "mse"


def val_loss(target: str, y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if is_cls(target):
        return E.binary_scores(y, p)["logloss"]
    return float(np.mean((y - p) ** 2))


def tune(target: str, cols, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    best, best_l = None, 9e9
    for prm in PARAM_GRID:
        m = make(target, cols, prm).fit(fit_df, fit_df[TARGETS[target]].values)
        l = val_loss(target, val_df[TARGETS[target]].values, m.predict(val_df))
        log("   tune", target, prm["max_depth"], prm["max_iter"], prm["min_samples_leaf"], round(l, 5))
        if l < best_l:
            best, best_l = prm, l
    return best


def metrics(target: str, y, p) -> dict:
    return E.binary_scores(y, p) if is_cls(target) else E.oktas_scores(y, p)


def per_row_loss(target: str, y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (p - y) ** 2 if is_cls(target) else np.abs(p - y)


def headline(target: str, d: pd.DataFrame, params: dict) -> dict:
    dd = rows_for(d, target)
    y_name = TARGETS[target]
    tr = dd[(dd.t < TEST_START) & dd.lead_h.isin(SHORT)]
    te = dd[(dd.t >= TEST_START) & (dd.t < TEST_END) & dd.lead_h.isin(CORE)]
    log(f"[{target}] headline: train {len(tr)}, test {len(te)} rows, {te.wmo.nunique()} stations, {te.night.nunique()} nights")
    y = te[y_name].values
    models = {CLIM: Clim("station_month").fit(tr, tr[y_name]), CLIM_M: Clim("month").fit(tr, tr[y_name]),
              RAW: raw_model(target).fit(tr, tr[y_name]),
              V1: make(target, F.V1_COLS, params[V1]).fit(tr, tr[y_name]),
              V2: make(target, F.v2_cols(), params[V2]).fit(tr, tr[y_name])}
    if target == "oktas":
        models["climatology median (station x month)"] = Clim("station_month", agg="median").fit(tr, tr[y_name])
        models["plain forecast (uncalibrated: cloud % / 100 x 8)"] = RawOktas()
    preds = {k: m.predict(te) for k, m in models.items()}
    out = {"n": int(len(te)), "stations": int(te.wmo.nunique()), "nights": int(te.night.nunique()),
           "train_rows": int(len(tr)), "models": {k: metrics(target, y, p) for k, p in preds.items()}}
    loss = {k: per_row_loss(target, y, p) for k, p in preds.items()}
    names = list(preds)
    pairs = [(V2, V1), (V2, RAW), (V2, CLIM), (V1, RAW), (V1, CLIM), (RAW, CLIM)]
    boot = E.bootstrap(te.night.values, loss, pairs)
    out["ci_" + ("brier" if is_cls(target) else "mae")] = boot["ci"]
    out["diff"] = boot["diff"]
    out["n_nights_bootstrap"] = boot["n_nights"]
    if is_cls(target):
        ll = {k: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4))) for k, p in preds.items()}
        b2 = E.bootstrap(te.night.values, ll, [(V2, V1), (V2, RAW)])
        out["ci_logloss"] = b2["ci"]
        out["diff_logloss"] = b2["diff"]
        out["reliability"] = {k: E.reliability_points(y, preds[k]) for k in (V2, V1, RAW)}
    else:
        out["reliability"] = {k: reliability_oktas(y, preds[k]) for k in (V2, V1, RAW)}
    # dusk and dawn hours (leads 6 and 18) as an extra check
    te2 = dd[(dd.t >= TEST_START) & (dd.t < TEST_END) & dd.lead_h.isin((6, 18))]
    if len(te2) > 200:
        out["dusk_dawn"] = {"n": int(len(te2)), **{k: metrics(target, te2[y_name].values, models[k].predict(te2)) for k in (CLIM, RAW, V1, V2)}}
    out["_models"] = models
    return out


def reliability_oktas(y, p, bins: int = 8) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 8, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"p": round(float(p[idx == b].mean()), 3), "obs": round(float(y[idx == b].mean()), 3), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]


def ablation(target: str, d: pd.DataFrame, params: dict) -> list[dict]:
    dd = rows_for(d, target)
    y_name = TARGETS[target]
    tr = dd[(dd.t < TEST_START) & dd.lead_h.isin(SHORT)]
    te = dd[(dd.t >= TEST_START) & (dd.t < TEST_END) & dd.lead_h.isin(CORE)]
    y = te[y_name].values
    configs = []
    for name in F.GROUPS:
        configs.append((f"cumulative: {name}", F.cols_for(name)))
    configs.append(("v2 with dust/aerosol at the valid time (hindsight, flatters)", F.cols_for("+ site (lat, lon)", hindsight_aerosol=True)))
    full = F.v2_cols()
    for g, c in list(F.GROUPS.items())[1:]:
        configs.append((f"v2 without {g[2:]}", [x for x in full if x not in c]))
    # v1 inputs plus one group on its own (isolates what each group adds by itself)
    for g, c in list(F.GROUPS.items())[1:]:
        configs.append((f"v1 inputs {g}", F.V1_COLS + c))
    base_loss = None
    out = []
    for label, cols in configs:
        m = make(target, cols, params[V2]).fit(tr, tr[y_name])
        p = m.predict(te)
        row = {"config": label, "n_inputs": len(cols), **metrics(target, y, p)}
        if label == "cumulative: v1 inputs":
            base_loss = per_row_loss(target, y, p)
            row["delta_vs_v1_inputs"] = 0.0
        elif base_loss is not None:
            diff = per_row_loss(target, y, p)
            b = E.bootstrap(te.night.values, {"a": diff, "b": base_loss}, [("a", "b")], reps=1000)
            dd_ = b["diff"]["a-b"]
            row["delta_vs_v1_inputs"] = dd_["point"]
            row["delta_ci"] = dd_["ci"]
        out.append(row)
        log(f"   ablation [{target}] {label}: {row.get('brier', row.get('mae')):.4f}")
    return out


def region_holdout(target: str, d: pd.DataFrame, params: dict) -> dict:
    dd = rows_for(d, target)
    y_name = TARGETS[target]
    res = {}
    for reg in ["catalonia", "south", "north", "canary"]:
        tr = dd[(dd.t < TEST_START) & dd.lead_h.isin(SHORT) & (dd.region != reg)]
        te = dd[(dd.t >= TEST_START) & (dd.t < TEST_END) & dd.lead_h.isin(CORE) & (dd.region == reg)]
        if len(te) < 150 or len(tr) < 500:
            continue
        y = te[y_name].values
        r = {"test_rows": int(len(te)), "test_stations": int(te.wmo.nunique()), "nights": int(te.night.nunique())}
        ms = {CLIM_M: Clim("month").fit(tr, tr[y_name]), RAW: raw_model(target).fit(tr, tr[y_name]),
              V1: make(target, F.V1_COLS, params[V1]).fit(tr, tr[y_name]), V2: make(target, F.v2_cols(), params[V2]).fit(tr, tr[y_name])}
        preds = {k: m.predict(te) for k, m in ms.items()}
        r["models"] = {k: metrics(target, y, p) for k, p in preds.items()}
        loss = {k: per_row_loss(target, y, p) for k, p in preds.items()}
        b = E.bootstrap(te.night.values, loss, [(V2, V1), (V2, RAW)], reps=1000)
        r["diff"] = b["diff"]
        # the same region when it was seen in training (how much the hold-out costs)
        trin = dd[(dd.t < TEST_START) & dd.lead_h.isin(SHORT)]
        seen = make(target, F.v2_cols(), params[V2]).fit(trin, trin[y_name])
        r["models"][V2 + " [region seen in training]"] = metrics(target, y, seen.predict(te))
        res[reg] = r
        log(f"   region [{target}] {reg}: " + ", ".join(f"{k}={v.get('brier', v.get('mae')):.4f}" for k, v in r["models"].items()))
    return res


def lead_curve(target: str, d: pd.DataFrame, params: dict) -> dict:
    """Models trained on every lead (with lead_h as an input), tested at 00 UT valid time by lead."""
    dd = rows_for(d, target)
    y_name = TARGETS[target]
    cols1, cols2 = F.V1_COLS + F.LEAD, F.v2_cols(lead=True)

    def fit_all(upto):
        tr = dd[dd.t < upto]
        return {"v1": make(target, cols1, params[V1]).fit(tr, tr[y_name]), "v2": make(target, cols2, params[V2]).fit(tr, tr[y_name]),
                "raw": raw_model(target).fit(tr, tr[y_name]), "clim": Clim("station_month").fit(tr, tr[y_name])}
    mv, mf = fit_all(VAL_START), fit_all(TEST_START)
    va = dd[(dd.t >= VAL_START) & (dd.t < TEST_START) & (dd.t.dt.hour == 0)]
    te = dd[(dd.t >= TEST_START) & (dd.t < TEST_END) & (dd.t.dt.hour == 0)]
    leads = sorted(te.lead_h.unique())
    rows = []
    for L in leads:
        a, b = te[te.lead_h == L], va[va.lead_h == L]
        if len(a) < 100:
            continue
        y = a[y_name].values
        pr = {"v2": mf["v2"].predict(a), "v1": mf["v1"].predict(a), "raw": mf["raw"].predict(a), "clim": mf["clim"].predict(a)}
        row = {"lead_h": int(L), "n": int(len(a)), "nights": int(a.night.nunique())}
        for k, p in pr.items():
            row[k] = metrics(target, y, p)
        # blend weight of the model against climatology, chosen on 2025 and applied to 2026
        if len(b) >= 100:
            yb = b[y_name].values
            pv, pc = mv["v2"].predict(b), mv["clim"].predict(b)
            ws = np.linspace(0, 1, 21)
            sc = [np.mean((w * pv + (1 - w) * pc - yb) ** 2) for w in ws]
            w = float(ws[int(np.argmin(sc))])
            row["best_weight"] = w
            row["blend"] = metrics(target, y, w * pr["v2"] + (1 - w) * pr["clim"])
        rows.append(row)
        log(f"   lead [{target}] {L}: " + ", ".join(f"{k}={row[k].get('brier', row[k].get('mae')):.4f}" for k in pr))
    return {"rows": rows, "validation_year": 2025}


def fit_final(d: pd.DataFrame, params: dict, version: str) -> dict:
    """Final models on every row (all leads, 2023-2026-10), as v1's final fit did."""
    bundle = {"version": version, "cols": F.v2_cols(lead=True)}
    for target in ("clear", "oktas", "haze"):
        dd = rows_for(d, target)
        m = make(target, F.v2_cols(lead=True), params[target][V2]).fit(dd, dd[TARGETS[target]].values)
        bundle[target] = m
        log("final fit", target, len(dd))
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--skip", default="", help="comma list of parts to skip: ablation,region,lead")
    ap.add_argument("--targets", default="clear,haze,oktas")
    a = ap.parse_args()
    d = load_table(a.table)
    log("table", len(d), "rows", d.wmo.nunique(), "stations", d.t.min(), d.t.max())
    skip = set(a.skip.split(","))
    res = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "table": {"rows": int(len(d)), "stations": int(d.wmo.nunique()), "first": str(d.t.min()), "last": str(d.t.max())}}
    all_params = {}
    for target in a.targets.split(","):
        dd = rows_for(d, target)
        trs = dd[(dd.t < TEST_START) & dd.lead_h.isin(SHORT)]
        fit_df, val_df = trs[trs.t < VAL_START], trs[trs.t >= VAL_START]
        log(f"[{target}] tuning on {len(fit_df)} fit / {len(val_df)} validation (2025) rows")
        params = {V1: tune(target, F.V1_COLS, fit_df, val_df), V2: tune(target, F.v2_cols(), fit_df, val_df)}
        all_params[target] = params
        r = {"params": params, "headline": None}
        h = headline(target, d, params)
        models = h.pop("_models")
        r["headline"] = h
        if "ablation" not in skip:
            r["ablation"] = ablation(target, d, params)
        if "region" not in skip:
            r["region_holdout"] = region_holdout(target, d, params)
        if "lead" not in skip:
            r["lead_curve"] = lead_curve(target, d, params)
        res[target] = r
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    if a.model_out:
        bundle = fit_final(d, all_params, "v2 (trained 2023-01..2026-10)")
        pickle.dump(bundle, open(a.model_out, "wb"))
        log("saved", a.model_out)


if __name__ == "__main__":
    from sky_tonight.quality.train_quality import main as _m      # so pickled classes are importable by name
    _m()
