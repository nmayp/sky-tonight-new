"""Score the clear-sky model on the Tenerife all-sky frames (AllSkyKamera ASK076 Deep Sky Teide) using real NOAA GFS
forecasts issued BEFORE each night, not after the fact. Nothing here is tuned: the model and the labels are fixed
before scoring.

Which forecast is used for a frame (main score): the 12 UT GFS cycle of the night's start date (the forecast a
person could read that afternoon), leads 8-18 h for frames at 20, 22, 00, 02, 04 and 06 UT.
Sensitivity (second score): the rule used for Apache Point in score_apo.py, the 18 UT cycle of the UT day before the
frame's own UT date. For frames after midnight UT that is the evening-before forecast (leads 6-12 h); for the
20 and 22 UT frames it is 26-28 h ahead.

Reads validation/tenerife/labels.json (my reading by eye: clear / partly / cloudy / unreadable), writes
validation/tenerife/labels_scored.json and prints the scores.
Run from the repository root:  python validation/score_tenerife.py"""
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sky_tonight import features, get_site, gfs

HERE = Path("validation/tenerife")
site = get_site("deep_sky_teide")
L = [o for o in json.load(open(HERE / "labels.json")) if o["label"] != "unreadable"]
SC = {"clear": 1.0, "partly": 0.5, "cloudy": 0.0}
bundle = pickle.load(open("data/model.pkl", "rb"))


def cycle_for(o, rule):
    t = pd.Timestamp(o["utc"]).round("h").to_pydatetime()
    if rule == 0:   # 12 UT of the night's start date
        return t, datetime.strptime(o["night"], "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=12)
    return t, (t.replace(hour=0, minute=0) - timedelta(days=1)).replace(hour=18)   # as score_apo.py


cache_f = HERE / "gfs_points.json"
cache = json.load(open(cache_f)) if cache_f.exists() else {}


def key(c, t):
    return f"{c:%Y%m%dT%H}|{t:%Y%m%dT%H}"


def fetch(job):
    c, t = job
    if key(c, t) in cache:
        return
    for attempt in range(5):   # the connection to the bucket drops now and then
        try:
            cache[key(c, t)] = gfs.to_model_features(gfs.point_forecast(c, int((t - c).total_seconds() // 3600),
                                                                          site["lat"], site["lon"]))
            json.dump(cache, open(cache_f, "w"))
            return
        except (requests.RequestException, OSError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"no forecast for {key(c, t)}")


def predict(rule):
    jobs = sorted({cycle_for(o, rule)[::-1] for o in L})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch, jobs))
    rows = []
    for o in L:
        t, c = cycle_for(o, rule)
        rows.append({"valid": pd.Timestamp(t), "lead_h": int((t - c).total_seconds() // 3600), **cache[key(c, t)]})
    F = pd.DataFrame(rows)
    x = features.build(F.set_index("valid")[features.WEATHER], site)
    return F.assign(p_model=np.asarray(bundle["model"].predict(x)), p_raw=np.asarray(bundle["raw"].predict(x)))


def report(name, d):
    out = {"n": len(d), "nights": int(d.night.nunique())}
    for k in ("p_model", "p_raw"):
        out["brier_" + k[2:]] = round(float(((d[k] - d.y) ** 2).mean()), 3)
    out["brier_constant"] = round(float(((d.y.mean() - d.y) ** 2).mean()), 3)
    sure = d[d.label != "partly"]
    out["sure"] = len(sure)
    # A hit: "clear" forecast (probability shown as 50% or more, whole percent as on the page) on a clear frame, or
    # "not clear" on a cloudy frame. The model's output sits on a plateau at about 50%, so rounding matters.
    out["hits_model"] = int(((sure.p_model.round(2) >= 0.5) == (sure.label == "clear")).sum())
    out["hits_raw"] = int(((sure.p_raw.round(2) >= 0.5) == (sure.label == "clear")).sum())
    print(name, out)
    return out


res = {}
for rule, name in ((0, "main (12 UT cycle of the night's start date)"), (1, "sensitivity (score_apo rule)")):
    P = predict(rule)
    d = pd.concat([pd.DataFrame(L).assign(y=lambda z: z.label.map(SC)), P.drop(columns="valid")], axis=1)
    d["cloud"] = d.cloud_cover
    res[rule] = (d, report(name, d))
json.dump(cache, open(HERE / "gfs_points.json", "w"))
d = res[0][0]

# Rank check among the clear / cloudy frames (0.5 = no better than chance, 1 = perfect); a plain diagnostic.
from sklearn.metrics import roc_auc_score  # noqa: E402

sure = d[d.label != "partly"]
auc = {k: round(float(roc_auc_score(sure.label == "clear", v)), 2)
       for k, v in (("model", sure.p_model), ("raw", sure.p_raw), ("forecast_cloud_inverted", -sure.cloud))}
print("AUC clear vs cloudy frames:", auc)

# Night-block bootstrap: frames of one night are not independent, so resample whole nights.
rng = np.random.default_rng(0)
nights = d.night.unique()
boot = {}
for other, lab in (("constant", "model minus constant"), ("p_raw", "model minus raw")):
    diffs = []
    for _ in range(2000):
        s = pd.concat([d[d.night == n] for n in rng.choice(nights, len(nights))])
        ref = ((s.y.mean() - s.y) ** 2).mean() if other == "constant" else ((s[other] - s.y) ** 2).mean()
        diffs.append(((s.p_model - s.y) ** 2).mean() - ref)
    print(f"{lab}: {np.mean(diffs):+.3f}, 95% of resamples {np.percentile(diffs, 2.5):+.3f} to {np.percentile(diffs, 97.5):+.3f}")
    boot[lab] = [round(float(np.mean(diffs)), 3), round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]

keep = ["file", "utc", "night", "label", "note", "sun_alt", "moon_alt", "moon_illum", "lead_h", "cloud", "p_model", "p_raw"]
json.dump(json.loads(d[keep].round(6).to_json(orient="records")), open(HERE / "labels_scored.json", "w"), indent=1)
json.dump({"main": res[0][1], "sensitivity_score_apo_rule": res[1][1], "auc_clear_vs_cloudy": auc,
           "bootstrap_by_night_mean_and_95pct": boot, "mean_p_model": round(float(d.p_model.mean()), 3),
           "mean_observed_score": round(float(d.y.mean()), 3)}, open(HERE / "scores.json", "w"), indent=1)
