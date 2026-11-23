"""Score the clear-sky model on the Apache Point all-sky frames using real NOAA GFS forecasts issued the evening
before (cycle 18 UT of the previous UT day, leads 7-18 h). Reproduces validation/apo/labels_scored.json.
Run from the repository root:  python validation/score_apo.py"""
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from sky_tonight import features, get_site, gfs

site = get_site("apache_point")
L = json.load(open("validation/apo/labels.json"))
SC = {"clear": 1.0, "partly": 0.5, "cloudy": 0.0}
jobs = set()
for o in L:
    t = pd.Timestamp(o["utc"]).round("h").to_pydatetime()
    cyc = (t.replace(hour=0, minute=0) - timedelta(days=1)).replace(hour=18)
    jobs.add((cyc, t))


def one(j):
    c, t = j
    lead = int((t - c).total_seconds() // 3600)
    return pd.Timestamp(t), lead, gfs.to_model_features(gfs.point_forecast(c, lead, site["lat"], site["lon"]))


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, sorted(jobs)))
F = pd.DataFrame([{"valid": t, "lead_h": l, **f} for t, l, f in res]).set_index("valid")
bundle = pickle.load(open("data/model.pkl", "rb"))
x = features.build(F[features.WEATHER], site)
F["p_model"], F["p_raw"] = bundle["model"].predict(x), bundle["raw"].predict(x)
d = pd.DataFrame([{**o, "y": SC[o["label"]], **F.loc[pd.Timestamp(o["utc"]).round("h")].to_dict()} for o in L])
for k in ("p_model", "p_raw"):
    print(k, "Brier", round(((d[k] - d.y) ** 2).mean(), 3))
print("constant Brier", round(((d.y.mean() - d.y) ** 2).mean(), 3))
