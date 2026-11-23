"""As-issued NOAA GFS forecasts (0.5 degree) at many points, with the extra fields the v2 model uses.

NOAA publishes every GFS run on AWS Open Data (bucket noaa-gfs-bdp-pds); each file has an index, so only the wanted
messages are downloaded by byte range. This is the forecast actually issued at the cycle time, for the given lead.
Compared with v1 this adds CAPE, surface visibility, boundary-layer height, precipitable water, winds at 250 and
500 hPa and relative humidity at 850 and 700 hPa.

Training rows: cycle 12 UT of day D, valid 21 UT, 00 UT, 03 UT (leads 9, 12, 15) for every day, plus valid 00 UT
at leads 36, 60, ..., 228 on every fourth day (for the lead-time curve and so the model knows how fast skill fades).
"""
from __future__ import annotations

import csv
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = {"User-Agent": "sky-tonight-research/0.3 (non-commercial research project; byte-range requests only)"}
RES = "0p50"

# (variable, level text in the index, our name)
WANT = [
    ("TCDC", "entire atmosphere", "cloud_cover"), ("LCDC", "low cloud layer", "cloud_cover_low"),
    ("MCDC", "middle cloud layer", "cloud_cover_mid"), ("HCDC", "high cloud layer", "cloud_cover_high"),
    ("TMP", "2 m above ground", "t2m_k"), ("DPT", "2 m above ground", "d2m_k"),
    ("UGRD", "10 m above ground", "u10"), ("VGRD", "10 m above ground", "v10"),
    ("PRES", "surface", "sp_pa"), ("PRATE", "surface", "prate"),
    ("CAPE", "surface", "cape"), ("VIS", "surface", "vis_m"), ("HPBL", "surface", "pbl_m"),
    ("PWAT", "entire atmosphere (considered as a single layer)", "pwat"),
    ("UGRD", "250 mb", "u250"), ("VGRD", "250 mb", "v250"), ("UGRD", "500 mb", "u500"), ("VGRD", "500 mb", "v500"),
    ("RH", "850 mb", "rh850"), ("RH", "700 mb", "rh700"),
]
RAW = [w[2] for w in WANT]
OUT_COLS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "relative_humidity_2m",
            "dew_point_2m", "temperature_2m", "wind_speed_10m", "surface_pressure", "precipitation",
            "cape", "vis_m", "pbl_m", "pwat", "wind_250", "wind_500", "shear_250_500", "rh850", "rh700"]


def url_for(cycle: datetime, fhour: int, res: str = RES) -> str:
    return f"{BASE}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.{res}.f{fhour:03d}"


def parse_index(text: str) -> list[tuple[int, str, str, str]]:
    rows = []
    for line in text.strip().splitlines():
        p = line.split(":")
        rows.append((int(p[1]), p[3], p[4], p[5]))      # byte offset, variable, level, step text
    return rows


def wanted_ranges(idx, fhour: int) -> dict[str, tuple[int, int | None]]:
    """Byte ranges of the wanted instantaneous messages (the 'N hour fcst' ones, not the 'ave' or 'acc' ones)."""
    out = {}
    for i, (off, var, lev, step) in enumerate(idx):
        for v, l, name in WANT:
            if var == v and lev == l and step == f"{fhour} hour fcst":
                out[name] = (off, idx[i + 1][0] - 1 if i + 1 < len(idx) else None)
    return out


class PointSet:
    """Inverse-distance weights over the four surrounding nodes of the 0.5 degree grid (same rule as v1)."""

    def __init__(self, points: dict[str, tuple[float, float]]):
        self.keys = list(points)
        self.points = points
        self._geom = None

    def _build(self, ni, nj, lat0, dlat, dlon, s2n):
        sign = 1 if s2n else -1
        J, I, W = [], [], []
        for k in self.keys:
            lat, lon = self.points[k]
            lon = lon % 360
            j0 = int(math.floor((lat - lat0) / (sign * dlat)))
            i0 = int(math.floor(lon / dlon))
            jj, ii, ww = [], [], []
            for dj in (0, 1):
                for di in (0, 1):
                    j, i = min(max(j0 + dj, 0), nj - 1), (i0 + di) % ni
                    la, lo = lat0 + sign * dlat * j, i * dlon
                    a = (math.sin(math.radians(la - lat) / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(la))
                         * math.sin(math.radians(lo - lon) / 2) ** 2)
                    d = 6371.0 * 2 * math.asin(math.sqrt(a))
                    jj.append(j); ii.append(i); ww.append(1.0 / max(d, 0.5))
            J.append(jj); I.append(ii); W.append(ww)
        W = np.array(W)
        self._geom = ((np.array(J) * ni + np.array(I)), W / W.sum(1, keepdims=True), (ni, nj, lat0, dlat, dlon, s2n))

    def sample(self, ec, gid) -> np.ndarray:
        ni, nj = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
        if self._geom is None:
            self._build(ni, nj, ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                        ec.codes_get(gid, "jDirectionIncrementInDegrees"), ec.codes_get(gid, "iDirectionIncrementInDegrees"),
                        ec.codes_get(gid, "jScansPositively") == 1)
        flat, w, _ = self._geom
        vals = ec.codes_get_values(gid)
        return (vals[flat] * w).sum(1)


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def raw_fields(cycle: datetime, fhour: int, pts: PointSet, res: str = RES) -> dict[str, np.ndarray]:
    """Raw GFS values at the points, by field name."""
    import eccodes as ec
    s = _session()
    url = url_for(cycle, fhour, res)
    r = s.get(url + ".idx", timeout=60)
    r.raise_for_status()
    rng = wanted_ranges(parse_index(r.text), fhour)
    missing = [n for n in RAW if n not in rng]
    if missing:
        raise RuntimeError(f"fields missing in {url}: {missing}")
    out = {}
    for name, (a, b) in rng.items():
        g = s.get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"}, timeout=120)
        g.raise_for_status()
        gid = ec.codes_new_from_message(g.content)
        try:
            out[name] = pts.sample(ec, gid)
        finally:
            ec.codes_release(gid)
    return out


def derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Model-ready columns (same units as v1 where they overlap: km/h, hPa, mm/h)."""
    t = raw["t2m_k"] - 273.15
    td = raw["d2m_k"] - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    w250 = 3.6 * np.hypot(raw["u250"], raw["v250"])
    w500 = 3.6 * np.hypot(raw["u500"], raw["v500"])
    return {
        "cloud_cover": raw["cloud_cover"], "cloud_cover_low": raw["cloud_cover_low"],
        "cloud_cover_mid": raw["cloud_cover_mid"], "cloud_cover_high": raw["cloud_cover_high"],
        "relative_humidity_2m": np.minimum(100.0, rh), "dew_point_2m": td, "temperature_2m": t,
        "wind_speed_10m": 3.6 * np.hypot(raw["u10"], raw["v10"]), "surface_pressure": raw["sp_pa"] / 100.0,
        "precipitation": raw["prate"] * 3600.0, "cape": raw["cape"], "vis_m": raw["vis_m"], "pbl_m": raw["pbl_m"],
        "pwat": raw["pwat"], "wind_250": w250, "wind_500": w500,
        "shear_250_500": 3.6 * np.hypot(raw["u250"] - raw["u500"], raw["v250"] - raw["v500"]),
        "rh850": raw["rh850"], "rh700": raw["rh700"],
    }


def job_list(start: str, end: str, long_every: int = 4, long_leads=range(36, 229, 24),
             short_leads=(9, 12, 15)) -> list[tuple[datetime, int]]:
    """(cycle, lead) pairs: 12 UT cycle each day, `short_leads` daily, long leads every `long_every` days."""
    d0, d1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    jobs, d, k = [], d0, 0
    while d <= d1:
        cyc = d.replace(hour=12, tzinfo=timezone.utc)
        jobs += [(cyc, lead) for lead in short_leads]
        if long_every and k % long_every == 0:
            jobs += [(cyc, lead) for lead in long_leads]
        d += timedelta(days=1)
        k += 1
    return jobs


def run_job(points: dict[str, tuple[float, float]], out_csv: str, start: str, end: str, workers: int = 8,
            short_leads=(9, 12, 15), long_every: int = 4) -> None:
    """Resumable: appends rows (wmo, valid_utc, cycle, lead_h, fields) and records finished (cycle, lead) pairs."""
    pts = PointSet(points)
    done_f = Path(out_csv + ".done")
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    jobs = [(c, h) for c, h in job_list(start, end, long_every=long_every, short_leads=short_leads) if f"{c:%Y%m%d%H}_{h}" not in done]
    new = not Path(out_csv).exists()
    fh = open(out_csv, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["wmo", "valid_utc", "cycle", "lead_h"] + OUT_COLS)

    def one(j):
        try:
            return j, derive(raw_fields(j[0], j[1], pts)), None
        except Exception as exc:                      # a missing cycle or a transient error: skip and go on
            return j, None, repr(exc)[:120]

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for (cyc, lead), res, err in ex.map(one, jobs):
            if res is not None:
                valid = cyc + timedelta(hours=lead)
                for i, k in enumerate(pts.keys):
                    w.writerow([k, valid.strftime("%Y-%m-%dT%H:00:00Z"), cyc.strftime("%Y-%m-%dT%H:00:00Z"), lead]
                               + [round(float(res[c][i]), 3) for c in OUT_COLS])
                fh.flush()
                with done_f.open("a") as f:
                    f.write(f"{cyc:%Y%m%d%H}_{lead}\n")
            else:
                print("skip", f"{cyc:%Y%m%d%H}_{lead}", err, file=sys.stderr, flush=True)
            n += 1
            if n % 50 == 0:
                print(n, "of", len(jobs), flush=True)
    fh.close()


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--short-leads", default="9,12,15", help="comma separated; use '6,18' with --long-every 0 for the dusk/dawn pass")
    ap.add_argument("--long-every", type=int, default=4, help="0 = no long leads")
    a = ap.parse_args()
    st = json.load(open(a.stations))
    run_job({s["wmo"]: (s["lat"], s["lon"]) for s in st}, a.out, a.start, a.end, a.workers,
            tuple(int(x) for x in a.short_leads.split(",")), a.long_every)
