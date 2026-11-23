"""Night-time SYNOP observations from Ogimet: total cloud (oktas), horizontal visibility and present weather.

Ogimet is a free service. One request covers a whole WMO block (08 = mainland Spain and the Balearics,
60 = Canary Islands) for a week, so the whole 2023-2026 history is a few hundred requests, made one at a time
with a pause. Only the synoptic hours at which cloud is reported are kept.

Code tables used (WMO 4377 visibility, 4677 present weather):
  VV 00-50 = 0.1-5.0 km, 56-80 = 6-30 km, 81-89 = 35-70 km, 90-96 = under 0.05 up to 4 km, 97 = 10 km, 98 = 20 km, 99 = 50 km or more.
  ww 04-09 = smoke, haze, dust; 10-12 = mist and shallow fog; 28 and 40-49 = fog; 30-35 = fog in the last hour.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import sys
import time
from pathlib import Path

import requests

UA = {"User-Agent": "sky-tonight-research/0.3 (non-commercial research project; low volume, one request at a time)"}
KEEP_HOURS = {18, 21, 0, 3, 6}
BLOCKS = ("08", "60")
HAZE_WW = set(range(4, 13)) | {28} | set(range(30, 36)) | set(range(40, 50))
COLS = ["wmo", "utc", "N", "vv_km", "ww", "wx_obs"]


def vv_km(code: int | None) -> float | None:
    """Horizontal visibility in km from the SYNOP VV code; None when not reported."""
    if code is None:
        return None
    if code == 0:
        return 0.05
    if code <= 50:
        return code / 10.0
    if 51 <= code <= 55:
        return None                                  # not used by the code table
    if code <= 80:
        return float(code - 50)
    if code <= 89:
        return 30.0 + (code - 80) * 5.0
    return {90: 0.03, 91: 0.05, 92: 0.2, 93: 0.5, 94: 1.0, 95: 2.0, 96: 4.0, 97: 10.0, 98: 20.0, 99: 50.0}.get(code)


def parse_line(line: str):
    """One Ogimet line -> dict (wmo, utc, N, vv_km, ww, wx_obs) or None.

    The line is 'block,YYYY,MM,DD,HH,MM,AAXX YYGGi IIiii iRiXhVV Nddff 1sTTT ... 7wwW1W2 ... 333 ...='.
    wx_obs is 1 when present weather was observed (the station reported it, or reported that none was significant;
    indicator iX 1, 4 and 7 = included, 2 and 5 = omitted because nothing significant, 3 and 6 = not observed)."""
    p = line.strip().rstrip("=").split(",")
    if len(p) < 7:
        return None
    try:
        y, mo, d, h = (int(p[i]) for i in (1, 2, 3, 4))
    except ValueError:
        return None
    toks = p[6].split()
    if "NIL" in toks or len(toks) < 5 or toks[0] != "AAXX":
        return None
    g3, g4 = toks[3], toks[4]
    if not (re.fullmatch(r"[0-9/]{5}", g3) and re.fullmatch(r"[0-9/]{5}", g4)):
        return None
    n = int(g4[0]) if g4[0].isdigit() and int(g4[0]) <= 8 else None     # 9 = sky obscured, / = not observed
    vv = int(g3[3:5]) if g3[3:5].isdigit() else None
    ix = int(g3[1]) if g3[1].isdigit() else None
    ww = None
    sec1 = []
    for t in toks[5:]:
        if t in ("333", "222", "444", "555"):
            break
        sec1.append(t)
    for t in sec1:
        if len(t) == 5 and t[0] == "7":
            ww = int(t[1:3]) if t[1:3].isdigit() else None
            break
    wx_obs = 1 if ix in (1, 2, 4, 5, 7) else 0
    if ww is None and ix in (2, 5):
        ww = 0                                       # reported as: nothing significant
    return {"wmo": p[0], "utc": f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:00:00Z", "N": n, "vv_km": vv_km(vv), "ww": ww,
            "wx_obs": wx_obs, "_hour": h}


def parse_text(text: str, keep_hours=KEEP_HOURS) -> list[dict]:
    out = []
    for line in text.splitlines():
        r = parse_line(line)
        if r is None or r["_hour"] not in keep_hours:
            continue
        del r["_hour"]
        out.append(r)
    return out


def poor_transparency(vv: float | None, ww: float | None, wx_obs: int) -> float:
    """1 = poor (visibility under 10 km, or haze/dust/smoke/mist/fog reported), 0 = good, nan = cannot tell."""
    bad_ww = ww is not None and int(ww) in HAZE_WW
    if (vv is not None and vv < 10.0) or bad_ww:
        return 1.0
    if vv is not None or (wx_obs and ww is not None):
        return 0.0
    return float("nan")


def fetch_window(block: str, d0: dt.date, d1: dt.date, retries: int = 3) -> str:
    """Raw Ogimet text for a block and date range. The service sometimes answers 200 with an empty body when the
    range is large, so an empty answer is retried and then the range is split in two."""
    url = f"https://www.ogimet.com/cgi-bin/getsynop?block={block}&begin={d0:%Y%m%d}0000&end={d1:%Y%m%d}2359"
    for k in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=180)
            if r.status_code == 200 and "AAXX" in r.text:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(6 * (k + 1))
    if d1 > d0:
        mid = d0 + (d1 - d0) // 2
        return fetch_window(block, d0, mid) + "\n" + fetch_window(block, mid + dt.timedelta(days=1), d1)
    raise RuntimeError(f"Ogimet did not answer for {block} {d0}")


def run_job(out_csv: str, start: str, end: str, window_days: int = 4, pause: float = 1.5) -> None:
    """Download week by week (resumable). One request at a time."""
    path = Path(out_csv)
    done_f = Path(out_csv + ".done")
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    new = not path.exists()
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    d, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while d <= last:
        e = min(d + dt.timedelta(days=window_days - 1), last)
        for block in BLOCKS:
            key = f"{block}_{d}"
            if key in done:
                continue
            try:
                rows = parse_text(fetch_window(block, d, e))
            except RuntimeError as exc:
                print("skip", key, exc, file=sys.stderr, flush=True)
                continue
            for r in rows:
                w.writerow(r)
            fh.flush()
            done_f.open("a").write(key + "\n")
            print(key, len(rows), flush=True)
            time.sleep(pause)
        d = e + dt.timedelta(days=1)
    fh.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-11-20")
    a = ap.parse_args()
    run_job(a.out, a.start, a.end)
