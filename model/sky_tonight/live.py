"""Live forecast -> clear-sky probability, logged and later scored against airport reports."""
from __future__ import annotations

import pickle
import sqlite3
import time

import pandas as pd
import requests

from . import SITE, features

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_site_log(
  site TEXT, issued_at INTEGER, hour INTEGER, lead_h REAL, p_model REAL, p_raw REAL, cloud_cover REAL,
  cloud_high REAL, source TEXT, PRIMARY KEY(site, issued_at, hour));
CREATE TABLE IF NOT EXISTS forecast_log(
  issued_at INTEGER, hour INTEGER, lead_h REAL, p_model REAL, p_raw REAL, cloud_cover REAL,
  source TEXT, PRIMARY KEY(issued_at, hour));
"""


def connect(path="data/live.sqlite") -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


MET_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
UA = {"User-Agent": "sky-tonight/0.1 (student project)"}


def _open_meteo(hours: int, site: dict) -> pd.DataFrame:
    r = requests.get(FORECAST_URL, timeout=60, params=dict(
        latitude=site["lat"], longitude=site["lon"], hourly=",".join(features.WEATHER),
        forecast_hours=hours, timezone="UTC"))
    r.raise_for_status()
    d = pd.DataFrame(r.json()["hourly"])
    d["time"] = pd.DatetimeIndex(d.time).tz_localize("UTC")
    return d.set_index("time")


def _met_norway(hours: int, site: dict) -> pd.DataFrame:
    """MET Norway: same variables under other names. Pressure is sea-level (the site is at 4 m)."""
    r = requests.get(MET_URL, headers=UA, timeout=60,
                     params=dict(lat=site["lat"], lon=site["lon"], altitude=int(site["elev_m"])))
    r.raise_for_status()
    rows = []
    for t in r.json()["properties"]["timeseries"][:hours]:
        i = t["data"]["instant"]["details"]
        rows.append({
            "time": pd.Timestamp(t["time"]),
            "cloud_cover": i["cloud_area_fraction"], "cloud_cover_low": i["cloud_area_fraction_low"],
            "cloud_cover_mid": i["cloud_area_fraction_medium"], "cloud_cover_high": i["cloud_area_fraction_high"],
            "relative_humidity_2m": i["relative_humidity"], "dew_point_2m": i["dew_point_temperature"],
            "temperature_2m": i["air_temperature"], "wind_speed_10m": i["wind_speed"] * 3.6,  # m/s -> km/h
            "surface_pressure": i["air_pressure_at_sea_level"],
            "precipitation": (t["data"].get("next_1_hours") or {}).get("details", {}).get("precipitation_amount", 0.0),
        })
    return pd.DataFrame(rows).set_index("time")


def fetch_forecast(hours: int = 48, site: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Hourly forecast and the name of the source that answered."""
    errors = []
    for name, fn in (("open-meteo", _open_meteo), ("met.no", _met_norway)):
        try:
            return fn(hours, site or SITE), name
        except Exception as exc:
            errors.append(f"{name}: {repr(exc)[:70]}")
    raise RuntimeError("; ".join(errors))


def load_model(path="data/model.pkl"):
    return pickle.load(open(path, "rb"))


def predict(forecast: pd.DataFrame, bundle, site: dict | None = None) -> pd.DataFrame:
    x = features.build(forecast, site)
    out = pd.DataFrame(index=x.index)
    out["sun_alt"] = x.sun_alt
    out["cloud_cover"] = x.cloud_cover
    out["cloud_high"] = x.cloud_cover_high
    out["p_model"] = bundle["model"].predict(x)
    out["p_raw"] = bundle["raw"].predict(x)
    return out


def log(con: sqlite3.Connection, pred: pd.DataFrame, source: str, issued_at: int | None = None) -> int:
    issued = issued_at or int(time.time())
    night = pred[pred.sun_alt < -6]
    rows = [(issued, int(t.timestamp()), (t.timestamp() - issued) / 3600, float(r.p_model),
             float(r.p_raw), float(r.cloud_cover), source)
            for t, r in night.iterrows() if t.timestamp() > issued]
    con.executemany("INSERT OR REPLACE INTO forecast_log VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def scoreboard(con: sqlite3.Connection) -> dict:
    """Score every logged forecast whose hour has passed and has an observation."""
    log_df = pd.read_sql("SELECT * FROM forecast_log", con)
    if log_df.empty:
        return {"n": 0}
    from . import synop
    obs = synop.recent(days=7)  # total cloud <= 2 oktas, the same target the model is trained on
    log_df["hour_ts"] = pd.to_datetime(log_df.hour, unit="s", utc=True)
    d = log_df.merge(obs[["clear"]], left_on="hour_ts", right_index=True, how="inner")
    if d.empty:
        return {"n": 0}
    out = {"n": int(len(d)), "nights": int(d.hour_ts.dt.floor("D").nunique())}
    for k in ("p_model", "p_raw"):
        out[f"brier_{k[2:]}"] = float(((d[k] - d.clear) ** 2).mean())
    out["clear_rate"] = float(d.clear.mean())
    out["mean_lead_h"] = float(d.lead_h.mean())
    return out


def log_all_sites(con: sqlite3.Connection, bundle, keys=None) -> dict:
    """Fetch the forecast for every place, convert it to clear-sky probabilities, and log it.
    The model was fitted on Barcelona; its use elsewhere is untested except where images check it."""
    from . import CHECK_SITES, SITES, get_site
    out = {}
    issued = int(time.time())
    for key in (keys or [*SITES, *CHECK_SITES]):
        st = get_site(key)
        try:
            fc, source = fetch_forecast(site=st)
            pred = predict(fc, bundle, st)
        except Exception as exc:
            out[key] = f"failed: {repr(exc)[:80]}"
            continue
        night = pred[pred.sun_alt < -6]
        rows = [(key, issued, int(t.timestamp()), (t.timestamp() - issued) / 3600, float(r.p_model),
                 float(r.p_raw), float(r.cloud_cover), float(r.cloud_high), source)
                for t, r in night.iterrows() if t.timestamp() > issued]
        con.executemany("INSERT OR REPLACE INTO forecast_site_log VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        out[key] = f"{len(rows)} night hours from {source}"
    return out
