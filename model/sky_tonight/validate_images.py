"""Pair logged all-sky camera images with the forecast made for that hour, and score the forecast.

Labels come from looking at each image (clear / partly / cloudy) and are stored in a labels CSV, so the
judgement is explicit and can be audited. clear = 1, partly = 0.5, cloudy = 0.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCORE = {"clear": 1.0, "partly": 0.5, "cloudy": 0.0}


def pair(index_csv: str, live_db: str, site: str = "roque", min_lead_h: float = 3.0) -> pd.DataFrame:
    """Each image with the forecast for its hour that was issued at least `min_lead_h` earlier."""
    im = pd.read_csv(index_csv, parse_dates=["utc"])
    im["hour_ts"] = im.utc.dt.tz_convert("UTC").dt.floor("h") if im.utc.dt.tz else im.utc.dt.tz_localize("UTC").dt.floor("h")
    con = sqlite3.connect(live_db)
    fc = pd.read_sql("SELECT * FROM forecast_site_log WHERE site=?", con, params=(site,))
    fc["hour_ts"] = pd.to_datetime(fc.hour, unit="s", utc=True)
    out = []
    for _, r in im.iterrows():
        c = fc[(fc.hour_ts == r.hour_ts) & (fc.lead_h >= min_lead_h)].sort_values("lead_h")
        if c.empty:
            continue
        f = c.iloc[0]  # the shortest lead that is still at least min_lead_h
        out.append({**r.to_dict(), "p_model": f.p_model, "p_raw": f.p_raw, "cloud_cover": f.cloud_cover,
                    "cloud_high": f.cloud_high, "lead_h": f.lead_h, "source": f.source})
    return pd.DataFrame(out)


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def evaluate(paired: pd.DataFrame, labels_csv: str) -> dict:
    lab = pd.read_csv(labels_csv)
    d = paired.merge(lab[["file", "label"]], on="file")
    d["y"] = d.label.map(SCORE)
    n = len(d)
    if n < 5:
        return {"n": n, "ok": False}
    res = {"n": n, "ok": True, "nights": int(pd.to_datetime(d.utc).dt.floor("D").nunique()),
           "share_clear": float((d.label == "clear").mean()),
           "brier_model": float(((d.p_model - d.y) ** 2).mean()),
           "brier_raw": float(((d.p_raw - d.y) ** 2).mean()),
           "brier_constant": float(((d.y.mean() - d.y) ** 2).mean()),
           "spearman_model": spearman(d.p_model, d.y), "spearman_raw": spearman(d.p_raw, d.y),
           "spearman_star_count": spearman(d.stars, d.y)}
    return res, d


def validation_json(d: pd.DataFrame, res: dict, img_dir: str, out: str, site_name: str):
    rows = []
    for _, r in d.sort_values("utc").iterrows():
        rows.append({"file": f"allsky/{r.file}", "local": pd.Timestamp(r.utc).strftime("%d %b %H:%M UTC"),
                     "p_model": round(float(r.p_model), 3), "cloud_total": round(float(r.cloud_cover)),
                     "label": r.label})
    json.dump({"status": "ready", "site": site_name, "camera": "Gran Telescopio CANARIAS all-sky camera",
               "nights": res["nights"], "images": rows, "summary": res.get("summary", "")},
              open(out, "w"), separators=(",", ":"))
