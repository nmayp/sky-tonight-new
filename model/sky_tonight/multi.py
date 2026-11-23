"""Multi-station training data: real GFS forecasts (issued 12 UT the same day) paired with SYNOP total cloud.

For each night D and each station, the forecast cycle is D 12 UT and the valid hours are 21 UT (lead 9 h), 00 UT
(lead 12 h) and 03 UT (lead 15 h): a forecast made in the afternoon for the evening and night ahead, which is how
the product would be used. Output rows are appended to a CSV so the job can stop and resume.
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import gfs

LEADS = (9, 12, 15)
FEATS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "relative_humidity_2m",
         "dew_point_2m", "temperature_2m", "wind_speed_10m", "surface_pressure", "precipitation"]


def run_job(stations_json: str, out_csv: str, start: str, end: str, step_days: int = 2, workers: int = 8,
            res: str = "0p50") -> None:
    st = json.load(open(stations_json))
    points = {s["wmo"]: (s["lat"], s["lon"]) for s in st}
    done_file = Path(out_csv + ".done")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    d0, d1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    jobs = []
    d = d0
    while d <= d1:
        cyc = d.replace(hour=12, tzinfo=timezone.utc)
        for lead in LEADS:
            key = f"{cyc:%Y%m%d%H}_{lead}"
            if key not in done:
                jobs.append((cyc, lead, key))
        d += timedelta(days=step_days)
    new = not Path(out_csv).exists()
    fh = open(out_csv, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["wmo", "valid_utc", "cycle", "lead_h"] + FEATS)

    def one(j):
        cyc, lead, key = j
        try:
            res_ = gfs.points_forecast(cyc, lead, points, res)
            return j, res_, None
        except Exception as exc:   # missing cycle or transient error: skip, keep going
            return j, None, repr(exc)[:80]

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for j, res_, err in ex.map(one, jobs):
            cyc, lead, key = j
            if res_ is not None:
                valid = cyc + timedelta(hours=lead)
                for wmo, v in res_.items():
                    f = gfs.to_model_features(v)
                    w.writerow([wmo, valid.strftime("%Y-%m-%dT%H:00:00Z"), cyc.strftime("%Y-%m-%dT%H:00:00Z"), lead]
                               + [round(f[k], 3) for k in FEATS])
                fh.flush()
                done_file.open("a").write(key + "\n")
            else:
                print("skip", key, err, file=sys.stderr, flush=True)
            n += 1
            if n % 20 == 0:
                print(n, "of", len(jobs), flush=True)
    fh.close()
