"""Feature groups, solar altitude, regions and time terms for the v2 sky-quality model."""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- feature groups (the ablation adds them one at a time) ---------------------------------------------------------
V1_WEATHER = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "relative_humidity_2m",
              "dew_point_2m", "temperature_2m", "wind_speed_10m", "surface_pressure", "precipitation"]
TIME_TERMS = ["dew_depression", "month_sin", "month_cos", "hour_sin", "hour_cos"]
V1_COLS = V1_WEATHER + TIME_TERMS + ["elev_km"]                       # what the deployed v1 model used
EXTRA_WEATHER = ["cape", "vis_m", "pbl_m", "pwat", "rh850", "rh700"]
AEROSOL = ["dust_0", "aod_0", "pm10_0"]                                # state 12 h before the cycle (no hindsight)
AEROSOL_HINDSIGHT = ["dust_v", "aod_v", "pm10_v"]                      # CAMS best-estimate at the valid time (flatters)
UPPER_WINDS = ["wind_250", "wind_500", "shear_250_500"]
SITE = ["lat", "lon"]
LEAD = ["lead_h"]

GROUPS = {
    "v1 inputs": V1_COLS,
    "+ extra weather": EXTRA_WEATHER,
    "+ dust/aerosol": AEROSOL,
    "+ upper winds": UPPER_WINDS,
    "+ site (lat, lon)": SITE,
}


def cols_for(upto: str, lead: bool = False, hindsight_aerosol: bool = False) -> list[str]:
    cols: list[str] = []
    for name, c in GROUPS.items():
        cols += c
        if name == upto:
            break
    if hindsight_aerosol:
        cols = [c for c in cols if c not in AEROSOL] + AEROSOL_HINDSIGHT
    return cols + (LEAD if lead else [])


def v2_cols(lead: bool = False) -> list[str]:
    return cols_for("+ site (lat, lon)", lead=lead)


# --- solar altitude (vectorised, low-precision formula good to about 0.02 degrees) -----------------------------------
def sun_altitude(t: pd.DatetimeIndex, lat, lon) -> np.ndarray:
    """Altitude of the Sun in degrees at UTC times t for latitude/longitude in degrees (no refraction)."""
    t = pd.DatetimeIndex(t)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    jd = t.to_julian_date().values if hasattr(t, "to_julian_date") else pd.DatetimeIndex(t).to_julian_date().values
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * n) % 360)
    ha = gmst + np.radians(lon) - ra
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


def add_time_terms(d: pd.DataFrame, t_col: str = "t") -> pd.DataFrame:
    d = d.copy()
    d["dew_depression"] = d["temperature_2m"] - d["dew_point_2m"]
    m, h = d[t_col].dt.month, d[t_col].dt.hour
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return d


def region(d: pd.DataFrame) -> pd.Series:
    """Same regions as v1's hold-out test."""
    r = pd.Series("interior/other", index=d.index)
    r[(d.lat < 38.5) & (d.lon > -7.5) & (d.lon < -1)] = "south"
    r[(d.lat > 42.3) & (d.lon < -1.0)] = "north"
    r[(d.lat > 40.4) & (d.lon > 0.15) & (d.lon < 3.4)] = "catalonia"
    r[d.wmo.astype(str).str.startswith("60")] = "canary"
    return r
