"""Sky-brightness model on the Stars4All TESS-W network (Spain), compared with a moon-and-climatology baseline.

Target: dimming = B - mag, where mag is the photometer's night-sky brightness (mag/arcsec2, larger = darker) and B
is that photometer's own dark baseline (90th percentile of its readings on moonless, forecast-clear nights of 2023-2025).
dimming > 0 means the sky is brighter than the place's own best. Also a yes/no version: dimming <= 0.5 mag
('the sky is within half a magnitude of its best').
Inputs: the same as the other v2 targets plus the Moon (height and lit fraction), which matters here and nowhere else.
Photometer readings are matched to the GFS values of the nearest SYNOP station within 35 km (0.5 degree cells are about
50 km wide, so neighbouring stations share cells); aerosol state from the Open-Meteo air-quality API at the photometer.

    python -m sky_tonight.quality.train_tess --tess work/tess_hourly.pkl --positions work/tess_positions.json \
        --gfs work/gfs_table.csv --stations data/quality/stations_spain_synop.json --out data/quality_tess_scores.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import aq, evaluate as E, features as F
from .models import PARAM_GRID, Boost, BoostReg, Clim
from .tess import km

warnings.filterwarnings("ignore")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-11-01", tz="UTC")
CORE = (9, 12, 15)
MAX_KM = 35.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def moon_features(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Topocentric Moon altitude (degrees) and lit fraction at UTC times."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time
    t = Time(times.tz_convert("UTC").tz_localize(None).to_pydatetime())
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)
    moon = get_body("moon", t, loc)
    alt = moon.transform_to(AltAz(obstime=t, location=loc)).alt.deg
    elong = get_body("moon", t).separation(get_sun(t)).rad          # geocentric, as the lit fraction needs
    return pd.DataFrame({"moon_alt": alt, "moon_lit": (1 - np.cos(elong)) / 2}, index=times)


def nearest_station(lat, lon, stations):
    best = min(stations, key=lambda s: km(lat, lon, s["lat"], s["lon"]))
    return best["wmo"], km(lat, lon, best["lat"], best["lon"])


def build(tess_pkl, positions_json, gfs_csv, stations_json) -> pd.DataFrame:
    hourly = pd.read_pickle(tess_pkl)
    pos = json.load(open(positions_json))
    stations = json.load(open(stations_json))
    match = {}
    for n, (la, lo) in pos.items():
        if n in hourly and len(hourly[n]):
            w, dkm = nearest_station(la, lo, stations)
            if dkm <= MAX_KM:
                match[n] = (w, dkm)
    log(len(match), "photometers within", MAX_KM, "km of a SYNOP station (of", len(pos), ")")
    need = sorted({w for w, _ in match.values()})
    parts = []
    for ch in pd.read_csv(gfs_csv, dtype={"wmo": str}, chunksize=500_000):
        ch["wmo"] = ch.wmo.str.zfill(5)
        parts.append(ch[ch.wmo.isin(need) & ch.lead_h.isin(CORE)])
    g = pd.concat(parts, ignore_index=True)
    g["t"] = pd.to_datetime(g.valid_utc, utc=True)
    aqh = aq.history({n: tuple(pos[n]) for n in match}, "2022-12-01", "2026-11-05")
    rows = []
    for n, (w, dkm) in match.items():
        s = hourly[n]
        s = s[(s > 8) & (s < 24.5)]
        gg = g[g.wmo == w].copy()
        if gg.empty or s.empty:
            continue
        before = s.reindex(gg.t - pd.Timedelta(hours=1)).values
        after = s.reindex(gg.t).values
        with np.errstate(all="ignore"):
            gg["mag"] = np.nanmean(np.vstack([before, after]), axis=0)
        gg = gg[gg.mag.notna()]
        if gg.empty:
            continue
        la, lo = pos[n]
        gg["sun_alt"] = F.sun_altitude(pd.DatetimeIndex(gg.t), la, lo)
        gg = gg[gg.sun_alt < -18].copy()
        if gg.empty:
            continue
        mf = moon_features(pd.DatetimeIndex(sorted(gg.t.unique())), la, lo)
        gg = gg.join(mf, on="t")
        gg["phot"], gg["lat"], gg["lon"] = n, la, lo
        gg["elev_km"] = 0.0
        h = aqh[n]
        cyc = pd.to_datetime(gg.cycle, utc=True).dt.normalize()
        gg["dust_0"] = h["dust"].reindex(pd.DatetimeIndex(cyc)).values
        gg["aod_0"] = h["aerosol_optical_depth"].reindex(pd.DatetimeIndex(cyc)).values
        gg["pm10_0"] = h["pm10"].reindex(pd.DatetimeIndex(cyc)).values
        rows.append(gg)
    d = pd.concat(rows, ignore_index=True)
    d = F.add_time_terms(d)
    d["night"] = (d.t - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    # per-photometer dark baseline from 2023-2025 only: moonless, forecast-clear nights
    tr = d[d.t < TEST_START]
    ok = tr[(tr.moon_alt < -5) & (tr.cloud_cover < 20)]
    base = ok.groupby("phot").mag.agg(lambda x: np.nan if len(x) < 30 else np.percentile(x, 90))
    d["base"] = d.phot.map(base)
    d = d[d.base.notna()].copy()
    d["dimming"] = d.base - d.mag
    d["dark_ok"] = (d.dimming <= 0.5).astype(float)
    d["wmo"] = d.phot
    d["moon_up_lit"] = np.where(d.moon_alt > 0, d.moon_lit, 0.0)
    d["moon_alt_pos"] = d.moon_alt.clip(lower=0)
    return d


MOON = ["moon_alt", "moon_lit", "moon_up_lit", "moon_alt_pos"]
SITE_B = ["base", "lat", "lon"]


def run(d: pd.DataFrame) -> dict:
    tr_all = d[d.t < TEST_START]
    te = d[(d.t >= TEST_START) & (d.t < TEST_END)]
    keep = tr_all.groupby("phot").size()
    keep = keep[keep >= 150].index.intersection(te.groupby("phot").size()[lambda s: s >= 40].index)
    tr_all, te = tr_all[tr_all.phot.isin(keep)], te[te.phot.isin(keep)]
    log("train", len(tr_all), "test", len(te), "photometers", len(keep), "nights test", te.night.nunique())
    fit_df, val_df = tr_all[tr_all.t < pd.Timestamp("2025-01-01", tz="UTC")], tr_all[tr_all.t >= pd.Timestamp("2025-01-01", tz="UTC")]
    base_cols = MOON + F.TIME_TERMS[1:] + SITE_B           # moon + calendar + the place's own baseline
    sets = {
        "moon + calendar + place baseline": base_cols,
        "+ weather (v1 inputs and extra)": base_cols + F.V1_WEATHER[:] + ["dew_depression"] + F.EXTRA_WEATHER,
        "+ dust/aerosol": base_cols + F.V1_WEATHER + ["dew_depression"] + F.EXTRA_WEATHER + F.AEROSOL,
        "+ upper winds": base_cols + F.V1_WEATHER + ["dew_depression"] + F.EXTRA_WEATHER + F.AEROSOL + F.UPPER_WINDS,
    }
    res = {"photometers": int(len(keep)), "train_rows": int(len(tr_all)), "test_rows": int(len(te)), "test_nights": int(te.night.nunique()),
           "baseline": "per-photometer 90th percentile of moonless forecast-clear nights, 2023-2025", "dimming": {}, "dark_ok": {}}
    # tuning on 2025 with the largest set (regression, squared error)
    from sklearn.ensemble import HistGradientBoostingRegressor
    best, bl = PARAM_GRID[0], 9e9
    for prm in PARAM_GRID:
        m = HistGradientBoostingRegressor(random_state=0, **prm).fit(fit_df[sets["+ upper winds"]], fit_df.dimming.values)
        l = float(np.mean((m.predict(val_df[sets["+ upper winds"]]) - val_df.dimming.values) ** 2))
        if l < bl:
            best, bl = prm, l
    res["params"] = best
    y = te.dimming.values
    loss, preds = {}, {}
    pc = Clim("station_month").fit(tr_all, tr_all.dimming)
    preds["climatology (photometer x month)"] = pc.predict(te)
    for name, cols in sets.items():
        m = HistGradientBoostingRegressor(random_state=0, **best).fit(tr_all[cols], tr_all.dimming.values)
        preds[name] = m.predict(te[cols])
        log("dimming", name, round(float(np.mean(np.abs(preds[name] - y))), 4))
    for k, p in preds.items():
        res["dimming"][k] = E.oktas_scores(y, p)
    for k, p in preds.items():
        loss[k] = np.abs(p - y)
    ks = list(sets)
    pairs = [(ks[i], ks[0]) for i in range(1, len(ks))] + [(ks[0], "climatology (photometer x month)")]
    b = E.bootstrap(te.night.values, loss, pairs, reps=1000)
    res["dimming_mae_ci"], res["dimming_mae_diff"] = b["ci"], b["diff"]
    # yes/no: sky within half a magnitude of the place's own best
    yb = te.dark_ok.values
    pb, lb = {}, {}
    pcb = Clim("station_month").fit(tr_all, tr_all.dark_ok)
    pb["climatology (photometer x month)"] = pcb.predict(te)
    for name, cols in sets.items():
        m = Boost(cols, best).fit(tr_all, tr_all.dark_ok.values)
        pb[name] = m.predict(te)
    for k, p in pb.items():
        res["dark_ok"][k] = E.binary_scores(yb, p)
        lb[k] = (np.clip(p, 1e-4, 1 - 1e-4) - yb) ** 2
    bb = E.bootstrap(te.night.values, lb, pairs, reps=1000)
    res["dark_ok_brier_ci"], res["dark_ok_brier_diff"] = bb["ci"], bb["diff"]
    res["dark_ok_base_rate"] = float(yb.mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--gfs", required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build(a.tess, a.positions, a.gfs, a.stations)
    d.to_csv(a.out.replace(".json", "_table.csv.gz"), index=False)
    res = run(d)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    log("saved", a.out)


if __name__ == "__main__":
    main()
