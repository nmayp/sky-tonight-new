"""Features for the clear-sky model, from a weather model's hourly output."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from . import SITE

WEATHER = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "relative_humidity_2m", "dew_point_2m", "temperature_2m", "wind_speed_10m",
    "surface_pressure", "precipitation",
]


def sun_altitude(times: pd.DatetimeIndex, site: dict | None = None) -> np.ndarray:
    site = site or SITE
    loc = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg, height=site["elev_m"] * u.m)
    t = Time(times.tz_localize(None).to_pydatetime(), scale="utc")
    return get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg


def build(weather: pd.DataFrame, site: dict | None = None) -> pd.DataFrame:
    """weather: index = UTC hour, columns include WEATHER. Adds derived columns."""
    x = weather[WEATHER].copy()
    x["dew_depression"] = x.temperature_2m - x.dew_point_2m
    idx = x.index
    x["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    x["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    x["sun_alt"] = sun_altitude(idx, site)
    return x


FEATURES = WEATHER + ["dew_depression", "month_sin", "month_cos", "hour_sin", "hour_cos"]
