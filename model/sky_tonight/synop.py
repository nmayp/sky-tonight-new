"""SYNOP total cloud cover (oktas) for Barcelona airport (WMO 08181) from Ogimet.

Ogimet is a free service; keep volume low (monthly requests, pauses). N is reported only on part of
the hours (manned/synoptic reports), so downstream code works with the hours that have it.
"""
from __future__ import annotations

import calendar
import re
import time

import pandas as pd
import requests

BLOCK = "08181"
UA = {"User-Agent": "sky-tonight-research/0.1 (student project; low volume)"}


def parse(text: str) -> pd.DataFrame:
    """Lines look like '08181,2026,11,20,15,00,AAXX 20154 08181 22/80 62404 ...='. N is the first digit
    of the 'Nddff' group (5th token); '/' means not observed, NIL means no report."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.rstrip("=").split(",")
        if len(parts) < 7:
            continue
        y, m, d, h = (int(parts[i]) for i in (1, 2, 3, 4))
        toks = parts[6].split()
        n = None
        if "NIL" not in toks and len(toks) >= 5 and re.fullmatch(r"[0-9/]{5}", toks[4]) and toks[4][0].isdigit():
            n = int(toks[4][0])
        rows.append((pd.Timestamp(year=y, month=m, day=d, hour=h, tz="UTC"), n))
    return pd.DataFrame(rows, columns=["hour", "N"]).drop_duplicates("hour")


def fetch(begin: str, end: str, retries: int = 3) -> pd.DataFrame:
    """begin/end as YYYYMMDDHHMM."""
    url = f"https://www.ogimet.com/cgi-bin/getsynop?block={BLOCK}&begin={begin}&end={end}"
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=90)
            if r.status_code == 200 and "AAXX" in r.text:
                return parse(r.text)
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    raise RuntimeError("Ogimet did not answer")


def fetch_history(months: list[tuple[int, int]], out: str = "data/synop_08181.csv", pause: float = 2.0):
    frames = []
    for y, m in months:
        last = calendar.monthrange(y, m)[1]
        frames.append(fetch(f"{y}{m:02d}010000", f"{y}{m:02d}{last:02d}2359"))
        time.sleep(pause)
    d = pd.concat(frames).drop_duplicates("hour").sort_values("hour")
    d.to_csv(out, index=False)
    return d


def recent(days: int = 7) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    d = fetch(start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M"))
    d = d.dropna(subset=["N"]).set_index("hour")
    d["clear"] = (d.N <= 2).astype(int)
    return d[["N", "clear"]]
