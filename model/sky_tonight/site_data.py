"""Numbers and maps for the project page: calibration, drift since training, skill by lead time, a Spain-wide climatology.

    python -m sky_tonight.site_data --synop S --feats F --stations J --decay D --grid G --gridfeats GF --out DIR
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time

from . import live_gfs
from .train_multi import EXTRA, WEATHER, Boost, Raw, load, score


def sun_alt(times: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Sun altitude (deg) for many places at many times: geocentric Sun once per time, then spherical trigonometry."""
    t = Time(times.tz_convert("UTC").tz_localize(None).values)
    s = get_sun(t)
    ra, dec = s.ra.rad, s.dec.rad
    gmst = t.sidereal_time("mean", "greenwich").rad
    out = np.empty((len(times), len(lat)))
    la = np.radians(lat)
    for i in range(len(times)):
        h = gmst[i] + np.radians(lon) - ra[i]
        out[i] = np.degrees(np.arcsin(np.sin(la) * np.sin(dec[i]) + np.cos(la) * np.cos(dec[i]) * np.cos(h)))
    return out


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            rows.append({"p": round(float(p[m].mean()), 3), "obs": round(float(y[m].mean()), 3), "n": int(m.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    for k in ["synop", "feats", "stations", "decay", "grid", "gridfeats", "out", "final"]:
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    d = load(a.synop, a.feats, a.stations)
    tr, te = d[d.t.dt.year < 2026], d[d.t.dt.year == 2026].copy()
    cols = WEATHER + EXTRA + ["elev_km"]
    m = Boost(cols).fit(tr, tr.clear.values)
    raw = Raw().fit(tr, tr.clear.values)
    clim = tr.groupby([tr.wmo, tr.t.dt.month]).clear.mean()
    te["clim"] = [clim.get((w, mo), tr.clear.mean()) for w, mo in zip(te.wmo, te.t.dt.month)]
    te["pm"], te["pr"] = m.predict(te), raw.predict(te)
    y = te.clear.values
    out = {"train_through": "2025-12-31", "test_from": "2026-01-01", "test_rows": int(len(te)),
           "reliability_model": reliability(y, te.pm.values), "reliability_raw": reliability(y, te.pr.values)}
    mon = []
    for mo, g in te.groupby(te.t.dt.month):
        mon.append({"month": int(mo), "n": int(len(g)), "model": round(float(np.mean((g.pm - g.clear) ** 2)), 4),
                    "raw": round(float(np.mean((g.pr - g.clear) ** 2)), 4), "clim": round(float(np.mean((g.clim - g.clear) ** 2)), 4)})
    out["monthly"] = mon
    out["overall"] = {"model": score(y, te.pm.values), "raw": score(y, te.pr.values), "clim_brier": float(np.mean((te.clim - y) ** 2))}
    # skill by forecast lead
    dd = load(a.synop, a.decay, a.stations)
    dd = dd[dd.t.dt.year == 2026].copy()
    dd["clim"] = [clim.get((w, mo), tr.clear.mean()) for w, mo in zip(dd.wmo, dd.t.dt.month)]
    dd["pm"], dd["pr"] = m.predict(dd), raw.predict(dd)
    lead = []
    for L, g in dd.groupby("lead_h"):
        yy = g.clear.values
        best = min(((float(np.mean((w * g.pm + (1 - w) * g.clim - yy) ** 2)), w) for w in np.linspace(0, 1, 21)))
        lead.append({"lead_h": int(L), "n": int(len(g)), "model": round(float(np.mean((g.pm - yy) ** 2)), 4),
                     "raw": round(float(np.mean((g.pr - yy) ** 2)), 4), "clim": round(float(np.mean((g.clim - yy) ** 2)), 4),
                     "best_weight": round(best[1], 2), "blend": round(best[0], 4),
                     "auc": round(float(score(yy, g.pm.values)["auc"]), 3)})
    out["lead"] = lead
    # climatology over a Spain-wide grid, from the final model fitted on all data
    bundle = pickle.load(open(a.final, "rb"))
    grid = pd.DataFrame(json.load(open(a.grid)))
    gf = pd.read_csv(a.gridfeats, dtype={"wmo": str}).drop_duplicates(["wmo", "valid_utc"])
    gf["t"] = pd.to_datetime(gf.valid_utc, utc=True)
    g = gf.merge(grid, on="wmo")
    parts = []
    for t_, gg in g.groupby("t"):
        parts.append(gg.assign(sun=sun_alt(pd.DatetimeIndex([t_]), gg.lat.values, gg.lon.values)[0]))
    g = pd.concat(parts)
    g = g[g.sun < -6].set_index("t")
    x = live_gfs.site_features(g.drop(columns=["sun"]), 0)
    x["elev_km"] = g.alt.values / 1000.0
    g["p"] = bundle["model"].predict(x)
    g["mon"] = g.index.month
    cl = g.groupby(["wmo", "mon"]).p.mean().unstack()
    cl = cl.reindex(grid.wmo)
    out_grid = {"lat": grid.lat.round(2).tolist(), "lon": grid.lon.round(2).tolist(), "alt": grid.alt.round().astype(int).tolist(),
                "clim": [[None if np.isnan(v) else int(round(v * 100)) for v in row] for row in cl.values]}
    json.dump(out_grid, open(f"{a.out}/grid.json", "w"), separators=(",", ":"))
    # does the grid climatology match what the stations saw?  (stations were in training, so this is a consistency check)
    st = {s["wmo"]: s for s in json.load(open(a.stations))}
    obs = d.groupby(["wmo", d.t.dt.month]).clear.mean()
    pairs = []
    for (w, mo), v in obs.items():
        s = st[w]
        k = ((grid.lat - s["lat"]) ** 2 + (grid.lon - s["lon"]) ** 2).idxmin()
        c = cl.iloc[k, mo - 1] if mo - 1 < cl.shape[1] and not np.isnan(cl.iloc[k, mo - 1]) else None
        if c is not None:
            pairs.append({"wmo": w, "month": int(mo), "obs": round(float(v), 3), "grid": round(float(c), 3)})
    pp = pd.DataFrame(pairs)
    out["clim_check"] = {"points": pairs, "r": round(float(np.corrcoef(pp.obs, pp.grid)[0, 1]), 3),
                         "mae": round(float((pp.obs - pp.grid).abs().mean()), 3)}
    out["trained_through"] = str(d.t.max().date())
    json.dump(out, open(f"{a.out}/skill.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["monthly", "lead"]}, indent=1)[:2500]); print(out["clim_check"]["r"], out["clim_check"]["mae"])


if __name__ == "__main__":
    from sky_tonight.site_data import main as _m
    _m()
