"""Skyline of every place in the list: how high the terrain rises above the flat horizon in each direction.

Elevation comes from the public Terrarium tiles (zoom 9, about 230 m a pixel in Spain; sources: SRTM, ASTER,
national datasets, see https://registry.opendata.aws/terrain-tiles/). For each place and each compass
direction (every 5 degrees, stored every 10 as the highest within 5 degrees either side) the terrain is sampled out to 45 km, and the skyline is the largest angle above
the horizontal, with the drop of the Earth's curve and a standard refraction coefficient (0.13). Buildings and
trees are not in the data, so a real skyline from a street can be higher.

Run from the model directory:  python -m sky_tonight.horizon places.json out.json
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Z = 9
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
STEP_DEG = 5
OUT_STEP = 10                   # stored every 10 degrees: the highest of the 5-degree values within +/-5 degrees
UNIT = 0.1                      # degrees per stored unit (uint8, so the skyline tops out at 25.5)
R_EFF = 6371.0 / (1 - 0.13)     # km, Earth's radius with refraction
DIST = np.concatenate([np.arange(0.3, 5, 0.3), np.arange(5, 15, 0.6), np.arange(15, 45.01, 1.2)])
CACHE = Path("/tmp/dem")


def tile_xy(lat, lon):
    n = 2 ** Z
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def fetch(x, y):
    f = CACHE / f"{Z}_{x}_{y}.png"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        for k in range(4):
            try:
                with urllib.request.urlopen(URL.format(z=Z, x=x, y=y), timeout=40) as r:
                    f.write_bytes(r.read())
                break
            except Exception:
                if k == 3:
                    return np.zeros((256, 256), np.float32)
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


class Mosaic:
    def __init__(self, lat0, lat1, lon0, lon1):
        xa, ya = tile_xy(lat1, lon0)
        xb, yb = tile_xy(lat0, lon1)
        self.x0, self.y0 = int(xa), int(ya)
        nx, ny = int(xb) - self.x0 + 1, int(yb) - self.y0 + 1
        jobs = [(self.x0 + i, self.y0 + j) for j in range(ny) for i in range(nx)]
        with ThreadPoolExecutor(8) as ex:
            tiles = list(ex.map(lambda t: fetch(*t), jobs))
        self.a = np.zeros((ny * 256, nx * 256), np.float32)
        for (tx, ty), t in zip(jobs, tiles):
            self.a[(ty - self.y0) * 256:(ty - self.y0 + 1) * 256, (tx - self.x0) * 256:(tx - self.x0 + 1) * 256] = t
        self.a = np.maximum(self.a, 0)   # the sea is flat

    def at(self, lat, lon):
        x, y = tile_xy(lat, lon)
        px = (x - self.x0) * 256 - 0.5
        py = (y - self.y0) * 256 - 0.5
        ix = np.clip(np.floor(px).astype(int), 0, self.a.shape[1] - 2)
        iy = np.clip(np.floor(py).astype(int), 0, self.a.shape[0] - 2)
        fx, fy = np.clip(px - ix, 0, 1), np.clip(py - iy, 0, 1)
        a = self.a
        return (a[iy, ix] * (1 - fx) * (1 - fy) + a[iy, ix + 1] * fx * (1 - fy) + a[iy + 1, ix] * (1 - fx) * fy + a[iy + 1, ix + 1] * fx * fy)


def skyline(m: Mosaic, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(places, 72) skyline angles in degrees."""
    az = np.radians(np.arange(0, 360, STEP_DEG))
    out = np.zeros((len(lat), len(az)), np.float32)
    h0 = m.at(lat, lon) + 2.0
    kmlat = 111.32
    for j, a in enumerate(az):
        best = np.zeros(len(lat), np.float32)
        for d in DIST:
            la = lat + d * math.cos(a) / kmlat
            lo = lon + d * math.sin(a) / (kmlat * np.cos(np.radians(lat)))
            h = m.at(la, lo)
            drop = d * d / (2 * R_EFF) * 1000.0
            ang = np.degrees(np.arctan2(h - h0 - drop, d * 1000.0))
            best = np.maximum(best, ang)
        out[:, j] = best
    return out


def main(places_file: str, out_file: str):
    places = json.load(open(places_file))["places"]
    lat = np.array([p[1] for p in places])
    lon = np.array([p[2] for p in places])
    canary = (lat < 31) & (lon < -12)
    sky = np.zeros((len(places), 360 // STEP_DEG), np.float32)
    for sel, box in ((~canary, (35.0, 44.5, -10.2, 4.8)), (canary, (27.3, 29.6, -18.5, -13.0))):
        if sel.any():
            m = Mosaic(*box)
            sky[sel] = skyline(m, lat[sel], lon[sel])
    pooled = np.maximum(np.maximum(sky, np.roll(sky, 1, axis=1)), np.roll(sky, -1, axis=1))[:, :: OUT_STEP // STEP_DEG]
    q = np.clip(np.round(pooled / UNIT), 0, 255).astype(np.uint8)
    json.dump({"step": OUT_STEP, "unit": UNIT, "n": len(places), "data": base64.b64encode(q.tobytes()).decode()}, open(out_file, "w"), separators=(",", ":"))
    print("places", len(places), "median top of skyline", float(np.median(sky.max(1))), "deg; share of places with >5 deg anywhere", float((sky.max(1) > 5).mean()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
