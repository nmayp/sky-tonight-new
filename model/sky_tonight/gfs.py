"""Real past and present forecasts from NOAA's GFS model, with no key and no daily limit.

NOAA publishes every GFS run on AWS Open Data (bucket noaa-gfs-bdp-pds). Each file has an index, so only the
messages we need are downloaded by byte range. This gives the forecast that was actually issued at a given
time (the 'cycle') for a given lead, not an after-the-fact reanalysis, and works for any point on Earth.
The variables are mapped to the same names the clear-sky model was trained on.
"""
from __future__ import annotations

import io
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = {"User-Agent": "sky-tonight-research/0.2 (student project; byte-range requests only)"}
# (variable, level text, our name). The instantaneous message is "<n> hour fcst" (not the "ave"/"acc" ones).
WANT = [
    ("TCDC", "entire atmosphere", "cloud_cover"), ("LCDC", "low cloud layer", "cloud_cover_low"),
    ("MCDC", "middle cloud layer", "cloud_cover_mid"), ("HCDC", "high cloud layer", "cloud_cover_high"),
    ("TMP", "2 m above ground", "t2m_k"), ("DPT", "2 m above ground", "d2m_k"),
    ("UGRD", "10 m above ground", "u10"), ("VGRD", "10 m above ground", "v10"),
    ("PRES", "surface", "sp_pa"), ("PRATE", "surface", "prate"),
]


def _url(cycle: datetime, fhour: int, res: str) -> str:
    d = f"{cycle:%Y%m%d}"
    return f"{BASE}/gfs.{d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.{res}.f{fhour:03d}"


def _index(url: str) -> list[tuple[int, str, str, str]]:
    r = requests.get(url + ".idx", headers=UA, timeout=60)
    r.raise_for_status()
    rows = []
    for line in r.text.strip().splitlines():
        p = line.split(":")
        rows.append((int(p[1]), p[3], p[4], p[5]))   # offset, var, level, step text
    return rows


def _ranges(idx, fhour: int):
    """Byte ranges of the wanted instantaneous messages."""
    out = {}
    for i, (off, var, lev, step) in enumerate(idx):
        for v, l, name in WANT:
            if var == v and lev == l and step == f"{fhour} hour fcst":
                end = idx[i + 1][0] - 1 if i + 1 < len(idx) else None
                out[name] = (off, end)
    return out


def point_forecast(cycle: datetime, fhour: int, lat: float, lon: float, res: str = "0p25") -> dict:
    """Values at the nearest grid points (inverse-distance weighted) for one cycle and one forecast hour."""
    import eccodes as ec
    url = _url(cycle, fhour, res)
    rng = _ranges(_index(url), fhour)
    vals = {}
    for name, (a, b) in rng.items():
        h = dict(UA, Range=f"bytes={a}-{'' if b is None else b}")
        r = requests.get(url, headers=h, timeout=120)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".grib2") as fh:
            fh.write(r.content)
            fh.flush()
            with open(fh.name, "rb") as f:
                gid = ec.codes_grib_new_from_file(f)
                near = ec.codes_grib_find_nearest(gid, lat, lon % 360 if lon < 0 else lon, npoints=4)
                ec.codes_release(gid)
        w = np.array([1.0 / max(p.distance, 0.5) for p in near])
        vals[name] = float(np.average([p.value for p in near], weights=w))
    return vals


def to_model_features(v: dict) -> dict:
    """Map GFS fields to the names and units the clear-sky model was trained on."""
    t = v["t2m_k"] - 273.15
    td = v["d2m_k"] - 273.15
    rh = 100 * math.exp(17.625 * td / (243.04 + td)) / math.exp(17.625 * t / (243.04 + t))
    return {
        "cloud_cover": v["cloud_cover"], "cloud_cover_low": v["cloud_cover_low"],
        "cloud_cover_mid": v["cloud_cover_mid"], "cloud_cover_high": v["cloud_cover_high"],
        "relative_humidity_2m": min(100.0, rh), "dew_point_2m": td, "temperature_2m": t,
        "wind_speed_10m": 3.6 * math.hypot(v["u10"], v["v10"]), "surface_pressure": v["sp_pa"] / 100.0,
        "precipitation": v["prate"] * 3600.0,   # kg m-2 s-1 -> mm per hour
    }


def forecast_series(cycle: datetime, valid_hours, lat: float, lon: float, res: str = "0p25") -> pd.DataFrame:
    """Model-ready features for the given valid times, all from one forecast cycle."""
    rows = {}
    for vt in valid_hours:
        lead = int((vt - cycle).total_seconds() // 3600)
        if lead < 1 or lead > 120:
            continue   # hourly files exist to f120; f000 has no cloud fields
        rows[pd.Timestamp(vt)] = to_model_features(point_forecast(cycle, lead, lat, lon, res))
    return pd.DataFrame.from_dict(rows, orient="index")


def _grid_values(ec, gid):
    ni, nj = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
    lat0, dlat = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"), ec.codes_get(gid, "jDirectionIncrementInDegrees")
    dlon = ec.codes_get(gid, "iDirectionIncrementInDegrees")
    south_to_north = ec.codes_get(gid, "jScansPositively") == 1
    vals = ec.codes_get_values(gid).reshape(nj, ni)
    return vals, lat0, dlat, dlon, south_to_north


def _nearest4(grid, lat, lon):
    """Inverse-distance average of the four surrounding grid nodes (what the training data used), vectorised."""
    vals, lat0, dlat, dlon, s2n = grid
    nj, ni = vals.shape
    lon = lon % 360
    sign = 1 if s2n else -1
    j0 = int(math.floor((lat - lat0) / (sign * dlat)))
    i0 = int(math.floor(lon / dlon))
    num = den = 0.0
    for dj in (0, 1):
        for di in (0, 1):
            j, i = min(max(j0 + dj, 0), nj - 1), (i0 + di) % ni
            la, lo = lat0 + sign * dlat * j, i * dlon
            a = math.sin(math.radians(la - lat) / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(la)) * math.sin(math.radians(lo - lon) / 2) ** 2
            d = 6371.0 * 2 * math.asin(math.sqrt(a))
            w = 1.0 / max(d, 0.5)
            num += w * float(vals[j, i]); den += w
    return num / den


def points_forecast(cycle: datetime, fhour: int, points: dict[str, tuple[float, float]],
                    res: str = "0p25") -> dict[str, dict]:
    """Same as point_forecast but for many places at once: each message is downloaded a single time."""
    import eccodes as ec
    url = _url(cycle, fhour, res)
    rng = _ranges(_index(url), fhour)
    out = {k: {} for k in points}
    for name, (a, b) in rng.items():
        h = dict(UA, Range=f"bytes={a}-{'' if b is None else b}")
        r = requests.get(url, headers=h, timeout=120)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".grib2") as fh:
            fh.write(r.content)
            fh.flush()
            with open(fh.name, "rb") as f:
                gid = ec.codes_grib_new_from_file(f)
                grid = _grid_values(ec, gid)
                for key, (lat, lon) in points.items():
                    out[key][name] = _nearest4(grid, lat, lon)
                ec.codes_release(gid)
    return {k: v for k, v in out.items() if len(v) == len(WANT)}
