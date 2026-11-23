"""Where to look tonight: clear-sky odds, Moon, darkness and events for several places in Spain."""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import SITES, events, get_site, live

NIGHTLIGHTS = "data/nightlights.json"
DARK_CLASSES = [(20, "very dark"), (40, "dark"), (70, "moderate"), (1e9, "bright")]


def darkness_class(v: float) -> str:
    return next(name for lim, name in DARK_CLASSES if v <= lim)


def moon_weight(malt: float, illum: float) -> float:
    """1 when the Moon is down; otherwise less the brighter it is. A simple, stated rule, not a measurement."""
    return 1.0 if malt <= 0 else max(0.0, 1.0 - illum)


def place_report(key: str, day: dt.date, bundle, tle, lights: dict, pred=None, source: str | None = None) -> dict:
    st = get_site(key)
    events.set_site(st)
    tz = ZoneInfo(st["tz"])
    if pred is None:
        fc, source = live.fetch_forecast(site=st)
        pred = live.predict(fc, bundle, st)
    times = events.night_window(day)
    sun = events.sun_alt(times)
    malt, illum = events.moon_info(times)
    hours = []
    for t, s_alt, ma, il in zip(times, sun, malt, illum):
        ts = pd.Timestamp(t)
        if s_alt >= -6 or ts not in pred.index:
            continue
        w = moon_weight(float(ma), float(il))
        hours.append({"local": ts.tz_convert(tz).strftime("%H:%M"), "utc": ts.strftime("%Y-%m-%dT%H:00Z"),
                      "p_clear": round(float(pred.p_model[ts]), 3), "p_raw": round(float(pred.p_raw[ts]), 3),
                      "cloud_total": round(float(pred.cloud_cover[ts])), "cloud_high": round(float(pred.cloud_high[ts])),
                      "moon_up": bool(ma > 0), "moon_alt": round(float(ma)), "moon_weight": round(w, 2)})
    good = round(sum(h["p_clear"] * h["moon_weight"] for h in hours), 2)
    ev = events.tonight(day, with_iss=True, tle=tle)
    items = []
    for e in ev["events"]:
        if e["kind"] in ("note",):
            continue
        items.append({"kind": e["kind"], "name": e["name"],
                      "when": e["peak_utc"].astimezone(tz).strftime("%H:%M") if e.get("peak_utc") else "",
                      "alt": None if "alt" not in e or np.isnan(e["alt"]) else round(e["alt"]),
                      "note": e.get("note", "")})
    nl = lights.get(key, {})
    d25 = nl.get("within_25km")
    return {"key": key, "name": st["name"], "elev_m": st["elev_m"], "tz": st["tz"], "forecast_source": source,
            "dark_hours": len(hours), "expected_good_hours": good,
            "night_lights_25km": d25, "darkness": darkness_class(d25) if d25 is not None else "unknown",
            "moon_illum": round(float(max(illum)), 2), "hours": hours, "events": items}


def _multi_predictions(bundle, day):
    """One GFS run for every place, scored with the multi-station model."""
    from . import live_gfs
    sites = {k: get_site(k) for k in SITES}
    hours = []
    for k, st in sites.items():
        events.set_site(st)
        hours += [h for h in events.night_window(day)]
    fcs, cycle = live_gfs.forecasts(sites, sorted(set(hours)))
    preds = {}
    for k, df in fcs.items():
        out = pd.DataFrame(index=df.index)
        out["p_model"] = bundle["model"].predict(df)
        out["p_raw"] = bundle["raw"].predict(df)
        out["cloud_cover"], out["cloud_high"] = df.cloud_cover, df.cloud_cover_high
        preds[k] = out
    return preds, f"NOAA GFS {cycle:%d %b %H}Z"


def build(day: dt.date | None = None, out: str = "site/planner.json") -> dict:
    import os
    multi = None
    if os.path.exists("data/model_multi.pkl"):
        import pickle
        multi = pickle.load(open("data/model_multi.pkl", "rb"))
    bundle = live.load_model()
    lights = json.load(open(NIGHTLIGHTS))
    tle = None
    try:
        tle = events.fetch_tle()
    except Exception:
        pass
    reports = []
    preds, msource = {}, None
    if multi is not None:
        try:
            d0 = day or dt.datetime.now(ZoneInfo(get_site("barcelona")["tz"])).date()
            preds, msource = _multi_predictions(multi, d0)
        except Exception as exc:
            print("multi-station forecast failed, using the single-site path:", repr(exc)[:100])
    for key in SITES:
        st = get_site(key)
        d = day or dt.datetime.now(ZoneInfo(st["tz"])).date()
        try:
            if key in preds:
                reports.append(place_report(key, d, multi, tle, lights, preds[key], msource))
            else:
                reports.append(place_report(key, d, bundle, tle, lights))
        except Exception as exc:
            reports.append({"key": key, "name": st["name"], "error": repr(exc)[:100]})
    ok = sorted([r for r in reports if "error" not in r], key=lambda r: -r["expected_good_hours"])
    res = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "places": ok,
           "failed": [r for r in reports if "error" in r]}
    json.dump(res, open(out, "w"), separators=(",", ":"), default=float)
    return res
