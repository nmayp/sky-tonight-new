"""Log the live all-sky camera at the Gran Telescopio Canarias (Roque de los Muchachos, La Palma).

The camera publishes one fresh image per minute and keeps no archive, so images are logged here, one an hour
while it is night, to compare with what the model predicted for that hour. Credit: Gran Telescopio CANARIAS
(GRANTECAN). Images are kept for validation of this project only.
"""
from __future__ import annotations

import csv
import email.utils
import io
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage

from . import events, get_site

URL = "https://www.gtc.iac.es/multimedia/netcam/camaraAllSky.jpg"
UA = {"User-Agent": "sky-tonight-validation/0.2 (student project; one image per hour)"}
SITE = get_site("roque")
COLS = ["utc", "file", "age_s", "sun_alt", "moon_alt", "moon_illum", "stars", "background"]
CENTER, ZENITH_R = (335, 240), 150   # fisheye centre and the zenith zone used for counting stars


def sky_state(when: datetime) -> tuple[float, float, float]:
    events.set_site(SITE)
    sun = float(events.sun_alt([when])[0])
    malt, illum = events.moon_info([when])
    return sun, float(malt[0]), float(illum[0])


def star_stats(img: Image.Image) -> tuple[int, float]:
    """Rough count of star-like peaks in the zenith zone, and the median background level (0-255).
    A clear dark sky gives many peaks; cloud, haze or a bright Moon gives few. It is a proxy, not a
    calibrated measurement."""
    g = np.asarray(img.convert("L"), dtype=float)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    zone = (xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2 < ZENITH_R ** 2
    sm = ndimage.gaussian_filter(g, 1.0)
    bg = ndimage.median_filter(sm, size=15)
    resid = sm - bg
    noise = 1.4826 * np.median(np.abs(resid[zone] - np.median(resid[zone]))) + 1e-6
    peak = (resid == ndimage.maximum_filter(resid, size=5)) & (resid > max(12.0, 6 * noise)) & zone
    return int(peak.sum()), float(np.median(g[zone]))


def capture(out_dir: str, force: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    sun, malt, illum = sky_state(now)
    if sun > -6 and not force:
        return {"skipped": f"daylight or twilight at the observatory (sun {sun:.0f} deg)"}
    r = requests.get(URL, headers=UA, timeout=40)
    r.raise_for_status()
    lm = email.utils.parsedate_to_datetime(r.headers["Last-Modified"]) if "Last-Modified" in r.headers else now
    age = (now - lm).total_seconds()
    if age > 20 * 60 and not force:
        return {"skipped": f"camera image is {age/60:.0f} minutes old"}
    img = Image.open(io.BytesIO(r.content))
    stars, bgl = star_stats(img)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"gtc_{lm:%Y%m%dT%H%MZ}.jpg"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=80)
    (out / name).write_bytes(buf.getvalue())
    idx = out / "index.csv"
    new = not idx.exists()
    with open(idx, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        w.writerow([lm.strftime("%Y-%m-%dT%H:%M:%SZ"), name, int(age), round(sun, 1), round(malt, 1),
                    round(illum, 3), stars, round(bgl, 1)])
    return {"saved": name, "age_s": int(age), "sun_alt": round(sun, 1), "moon_alt": round(malt, 1),
            "moon_illum": round(illum, 2), "stars": stars, "background": round(bgl, 1)}


# Ordinary webcams at the same observatory: they show the sky and horizon the way a person standing there would
# see them (a fixed lens, not a fisheye). Credit: Gran Telescopio CANARIAS (GRANTECAN).
WEBCAMS = {
    "gtc_ext": "https://www.gtc.iac.es/multimedia/netcam/camaraExt.jpg",
    "gtc_visual": "https://www.gtc.iac.es/multimedia/netcam/camaraExt3.jpg",
}
WEBCOLS = ["utc", "camera", "file", "age_s", "sun_alt", "moon_alt", "moon_illum"]


def capture_webcams(out_dir: str, force: bool = False, width: int = 900) -> dict:
    now = datetime.now(timezone.utc)
    sun, malt, illum = sky_state(now)
    if sun > -6 and not force:
        return {"skipped": f"daylight or twilight at the observatory (sun {sun:.0f} deg)"}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for cam, url in WEBCAMS.items():
        try:
            r = requests.get(url, headers=UA, timeout=40)
            r.raise_for_status()
            lm = email.utils.parsedate_to_datetime(r.headers["Last-Modified"]) if "Last-Modified" in r.headers else now
            age = (now - lm).total_seconds()
            if age > 20 * 60 and not force:
                saved[cam] = f"stale ({age/60:.0f} min)"
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img = img.resize((width, int(img.height * width / img.width)))
            name = f"{cam}_{lm:%Y%m%dT%H%MZ}.jpg"
            img.save(out / name, "JPEG", quality=78)
            idx = out / "index.csv"
            new = not idx.exists()
            with open(idx, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(WEBCOLS)
                w.writerow([lm.strftime("%Y-%m-%dT%H:%M:%SZ"), cam, name, int(age), round(sun, 1),
                            round(malt, 1), round(illum, 3)])
            saved[cam] = name
        except Exception as exc:
            saved[cam] = f"failed: {repr(exc)[:60]}"
    return saved
