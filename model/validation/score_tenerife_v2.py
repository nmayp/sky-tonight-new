"""Score the v2 clear-sky output on the same 52 Tenerife all-sky frames as score_tenerife.py, same rule (12 UT GFS
cycle of the night's start date), same label scores (clear 1, partly 0.5, cloudy 0). Nothing is tuned.
Run from the model directory:  PYTHONPATH=. python validation/score_tenerife_v2.py"""
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sky_tonight import get_site
from sky_tonight.quality import aq, features as F, gfs_table as G

site = get_site("deep_sky_teide")
L = [o for o in json.load(open("validation/tenerife/labels.json")) if o["label"] != "unreadable"]
SC = {"clear": 1.0, "partly": 0.5, "cloudy": 0.0}
bundle = pickle.load(open("data/quality_v2.pkl", "rb"))
pts = G.PointSet({"t": (site["lat"], site["lon"])})
hist = aq.history({"t": (site["lat"], site["lon"])}, "2026-11-01", "2026-11-23")["t"]
rows = []
for o in L:
    t = pd.Timestamp(o["utc"]).round("h")
    cyc = datetime.strptime(o["night"], "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=12)
    lead = int((t.to_pydatetime() - cyc).total_seconds() // 3600)
    lo, hi = (lead // 3) * 3, -(-lead // 3) * 3          # 0.5 degree files exist every 3 hours: interpolate in time
    a = {k: float(v[0]) for k, v in G.derive(G.raw_fields(cyc, lo, pts)).items()}
    b = a if hi == lo else {k: float(v[0]) for k, v in G.derive(G.raw_fields(cyc, hi, pts)).items()}
    w = 0.0 if hi == lo else (lead - lo) / 3.0
    d = {k: a[k] * (1 - w) + b[k] * w for k in a}
    d.update(lead_h=lead, t=t, lat=site["lat"], lon=site["lon"], elev_km=site["elev_m"] / 1000.0, y=SC[o["label"]], night=o["night"], utc=o["utc"], file=o.get("file"))
    base = pd.Timestamp(cyc).normalize()
    r0 = hist.reindex([base]).iloc[0]
    d.update(dust_0=r0.dust, aod_0=r0.aerosol_optical_depth, pm10_0=r0.pm10)
    rows.append(d)
D = F.add_time_terms(pd.DataFrame(rows))
D["p_v2"] = bundle["clear"].predict(D)
D["p_okta"] = bundle["oktas"].predict(D)
v1 = pickle.load(open("data/model.pkl", "rb"))
D["p_raw_v1cal"] = np.asarray(v1["raw"].predict(D.rename(columns={})))
const = D.y.mean()
res = {"n": int(len(D)), "nights": int(D.night.nunique()),
       "brier_v2": round(float(((D.p_v2 - D.y) ** 2).mean()), 3),
       "brier_calibrated_raw_forecast": round(float(((D.p_raw_v1cal - D.y) ** 2).mean()), 3),
       "brier_constant_guess": round(float(((const - D.y) ** 2).mean()), 3),
       "mean_p_v2": round(float(D.p_v2.mean()), 3), "mean_read": round(float(D.y.mean()), 3)}
print(json.dumps(res, indent=1))
json.dump(res, open("validation/tenerife/scores_v2.json", "w"), indent=1)
json.dump([{"utc": r.utc, "file": r.file, "p_v2": round(float(r.p_v2), 3)} for r in D.itertuples()], open("validation/tenerife/frames_v2.json", "w"), indent=0)
