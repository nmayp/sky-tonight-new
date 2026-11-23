"""Observed sky cover from airport METAR reports (Iowa Environmental Mesonet archive)."""
from __future__ import annotations

import pandas as pd

# Octas of the cloud layer; METAR groups are FEW(1-2) SCT(3-4) BKN(5-7) OVC(8).
OCTAS = {"NSC": 0.0, "SKC": 0.0, "CLR": 0.0, "FEW": 1.5, "SCT": 3.5, "BKN": 6.0, "OVC": 8.0, "VV": 8.0}
CLEAR_MAX_OCTAS = 2.0  # "clear" = at most a few clouds (FEW) in the reported layers


def load_metar(path: str) -> pd.DataFrame:
    """Hourly cloud cover in octas from a station's METAR archive.

    A report with no cloud groups at all is CAVOK (checked on a sample: every empty row
    was CAVOK), i.e. no cloud below 5000 ft, so it counts as 0 octas.
    METAR does not report thin high cloud, so 'clear' here means clear of low/mid cloud.
    """
    d = pd.read_csv(path, parse_dates=["valid"])
    cols = [c for c in ("skyc1", "skyc2", "skyc3", "skyc4") if c in d.columns]
    for c in cols:
        d[c] = d[c].astype(str).str.strip().map(OCTAS)  # 'M' -> NaN
    d["octas"] = d[cols].max(axis=1).fillna(0.0)
    d["hour"] = d.valid.dt.round("h")
    # keep the report closest to the full hour
    d["gap"] = (d.valid - d.hour).abs()
    d = d.sort_values("gap").drop_duplicates("hour").sort_values("hour")
    out = d[["hour", "octas"]].set_index("hour")
    out["clear"] = (out.octas <= CLEAR_MAX_OCTAS).astype(int)
    return out
