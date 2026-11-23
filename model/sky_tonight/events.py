"""What is in the sky tonight, from astronomy (no ML): Moon, planets, ISS, meteor showers."""
from __future__ import annotations

import datetime as dt
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun, solar_system_ephemeris
from astropy.time import Time
from sgp4.api import WGS72, Satrec, jday

from . import SITE

warnings.filterwarnings("ignore", message=".*NonRotationTransformation.*")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")
TZ = ZoneInfo(SITE["tz"])
LOC = EarthLocation(lat=SITE["lat"] * u.deg, lon=SITE["lon"] * u.deg, height=SITE["elev_m"] * u.m)


def set_site(site: dict) -> None:
    """Point every calculation in this module at another place (single-threaded use)."""
    global TZ, LOC
    TZ = ZoneInfo(site["tz"])
    LOC = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg, height=site["elev_m"] * u.m)
# Typical naked-eye brightness (not computed); Uranus/Neptune need a telescope.
PLANETS = {"mercury": "about 0 (varies -2..+3)", "venus": "about -4", "mars": "about +1 (varies -3..+1.8)",
           "jupiter": "about -2.5", "saturn": "about +0.8"}
# Typical peaks; check the IMO calendar for the exact year. (name, month, day, ZHR, RA deg, Dec deg)
SHOWERS = [("Quadrantids", 1, 3, 80, 230, 49), ("Lyrids", 4, 22, 18, 271, 34),
           ("Perseids", 8, 12, 100, 48, 58), ("Orionids", 10, 21, 20, 95, 16),
           ("Leonids", 11, 17, 15, 152, 22), ("Geminids", 12, 14, 150, 112, 33),
           ("Ursids", 12, 22, 10, 217, 76)]


def night_window(day: dt.date) -> list[dt.datetime]:
    """Hourly UTC times from 18:00 local on `day` to 07:00 next day."""
    start = dt.datetime.combine(day, dt.time(18), tzinfo=TZ).astimezone(dt.timezone.utc)
    return [start + dt.timedelta(hours=h) for h in range(14)]


def _altaz(body: str | SkyCoord, times: list[dt.datetime]):
    t = Time(times)
    frame = AltAz(obstime=t, location=LOC)
    if isinstance(body, str):
        with solar_system_ephemeris.set("builtin"):
            return get_body(body, t, LOC).transform_to(frame)
    return body.transform_to(frame)


def sun_alt(times):
    t = Time(times)
    return get_sun(t).transform_to(AltAz(obstime=t, location=LOC)).alt.deg


def moon_info(times):
    t = Time(times)
    with solar_system_ephemeris.set("builtin"):
        moon, sun = get_body("moon", t, LOC), get_sun(t)
    elong = moon.separation(sun).deg
    illum = (1 - np.cos(np.radians(elong))) / 2
    altaz = moon.transform_to(AltAz(obstime=t, location=LOC))
    return altaz.alt.deg, illum


def planets_tonight(times, dark):
    out = []
    for name, mag in PLANETS.items():
        aa = _altaz(name, times)
        alt = aa.alt.deg
        ok = dark & (alt > 10)
        if ok.any():
            i = int(np.argmax(np.where(ok, alt, -90)))
            out.append({"kind": "planet", "name": name.title(), "peak_utc": times[i], "alt": float(alt[i]),
                        "az": float(aa.az.deg[i]), "note": f"typical brightness {mag}",
                        "hours_utc": [times[j] for j in range(len(times)) if ok[j]]})
    return out


def fetch_tle(norad: int = 25544) -> tuple[str, str]:
    """Latest two-line element set; tries two public sources."""
    errors = []
    for url in (f"https://api.wheretheiss.at/v1/satellites/{norad}/tles",
                f"https://tle.ivanstanojevic.me/api/tle/{norad}"):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            j = r.json()
            return j["line1"], j["line2"]
        except Exception as exc:
            errors.append(repr(exc)[:60])
    raise RuntimeError("no TLE source reachable: " + "; ".join(errors))


def iss_passes(day: dt.date, tle: tuple[str, str] | None = None, step_s: int = 20):
    """Visible ISS passes: above 10 deg, observer in darkness (sun < -6), ISS in sunlight."""
    l1, l2 = tle or fetch_tle()
    sat = Satrec.twoline2rv(l1, l2, WGS72)
    start = dt.datetime.combine(day, dt.time(18), tzinfo=TZ).astimezone(dt.timezone.utc)
    secs = np.arange(0, 14 * 3600, step_s)
    times = [start + dt.timedelta(seconds=int(s)) for s in secs]
    jd = np.array([jday(t.year, t.month, t.day, t.hour, t.minute, t.second)[0] for t in times])
    fr = np.array([jday(t.year, t.month, t.day, t.hour, t.minute, t.second)[1] for t in times])
    err, r_km, _ = sat.sgp4_array(jd, fr)
    t = Time(times)
    from astropy.coordinates import TEME, ITRS, CartesianRepresentation
    teme = TEME(CartesianRepresentation(r_km.T * u.km), obstime=t)
    itrs = teme.transform_to(ITRS(obstime=t))
    aa = itrs.transform_to(AltAz(obstime=t, location=LOC))
    alt = aa.alt.deg
    sun = get_sun(t)
    sun_aa = sun.transform_to(AltAz(obstime=t, location=LOC)).alt.deg
    # cylindrical Earth-shadow test: ISS is sunlit unless behind Earth w.r.t. the Sun
    s_vec = sun.cartesian.xyz.to_value(u.km) if False else None
    sun_gcrs = sun.transform_to(TEME(obstime=t)).cartesian.xyz.value  # unit-less direction after normalise
    sun_dir = sun_gcrs / np.linalg.norm(sun_gcrs, axis=0)
    rv = r_km.T  # (3, N)
    along = (rv * sun_dir).sum(axis=0)
    perp = np.linalg.norm(rv - along * sun_dir, axis=0)
    sunlit = ~((along < 0) & (perp < 6371.0))
    vis = (alt > 10) & (sun_aa < -6) & sunlit & (err == 0)
    passes, i = [], 0
    while i < len(vis):
        if vis[i]:
            j = i
            while j + 1 < len(vis) and vis[j + 1]:
                j += 1
            k = i + int(np.argmax(alt[i:j + 1]))
            passes.append({"kind": "iss", "name": "ISS pass", "start_utc": times[i], "end_utc": times[j],
                           "peak_utc": times[k], "alt": float(alt[k]), "az": float(aa.az.deg[k]),
                           "note": f"{(j - i) * step_s // 60} min {(j - i) * step_s % 60:02d} s above 10 deg"})
            i = j + 1
        else:
            i += 1
    return passes


def meteor_showers(day: dt.date, times, dark):
    out = []
    for name, m, d, zhr, ra, dec in SHOWERS:
        peak = dt.date(day.year, m, d)
        off = (day - peak).days
        if abs(off) <= 3:
            aa = _altaz(SkyCoord(ra * u.deg, dec * u.deg), times)
            alt = aa.alt.deg
            ok = dark & (alt > 20)
            malt, illum = moon_info(times)
            glare = float(np.mean(illum[ok & (malt > 0)])) if (ok & (malt > 0)).any() else 0.0
            if ok.any():
                i = int(np.argmax(np.where(ok, alt, -90)))
                out.append({"kind": "meteors", "name": f"{name} (peak ~{peak:%d %b}, {off:+d} d)",
                            "peak_utc": times[i], "alt": float(alt[i]), "az": float(aa.az.deg[i]),
                            "note": f"typical ZHR {zhr}; moon glare {glare:.0%} lit when up; dates are typical, check IMO",
                            "hours_utc": [times[j] for j in range(len(times)) if ok[j]]})
    return out


def tonight(day: dt.date, with_iss: bool = True, tle=None) -> dict:
    times = night_window(day)
    sun = sun_alt(times)
    dark = sun < -6
    malt, illum = moon_info(times)
    events = planets_tonight(times, dark) + meteor_showers(day, times, dark)
    if with_iss:
        try:
            events += iss_passes(day, tle)
        except Exception as exc:  # the TLE source can be down; the rest still works
            events.append({"kind": "note", "name": "ISS passes unavailable", "note": repr(exc)[:80]})
    mi = int(np.argmax(malt))
    events.insert(0, {"kind": "moon", "name": f"Moon {illum[mi]:.0%} lit", "peak_utc": times[mi],
                      "alt": float(malt[mi]), "az": float("nan"),
                      "note": "bright Moon washes out faint objects" if illum[mi] > 0.5 else "dark-ish sky",
                      "hours_utc": [times[j] for j in range(len(times)) if malt[j] > 0]})
    return {"day": day.isoformat(), "dark_hours_utc": [times[i] for i in range(len(times)) if dark[i]],
            "moon_illum": float(illum[mi]), "events": events}
