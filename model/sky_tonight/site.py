"""Static page: tonight's events with the chance the sky lets you see them."""
from __future__ import annotations

import html
import math

import pandas as pd

from . import SITE
from .events import TZ


def _p_at(pred: pd.DataFrame, ts) -> float | None:
    t = pd.Timestamp(ts).tz_convert("UTC").floor("h")
    return float(pred.p_model.get(t)) if t in pred.index else None


def _label(p, high=None):
    if p is None:
        return "no forecast"
    extra = "" if high is None else f" (forecast high cloud {high:.0f}%)"
    return f"{p:.0%} chance of at most 2/8 cloud{extra}"


def _high_at(pred: pd.DataFrame, ts):
    t = pd.Timestamp(ts).tz_convert("UTC").floor("h")
    return float(pred.cloud_high.get(t)) if t in pred.index else None


def render(ev: dict, pred: pd.DataFrame, scoreboard: dict, offline: dict) -> str:
    rows = []
    for e in ev["events"]:
        if e["kind"] == "note":
            rows.append(f"<li class='muted'>{html.escape(e['name'])}: {html.escape(e['note'])}</li>")
            continue
        peak = e["peak_utc"].astimezone(TZ).strftime("%H:%M")
        p = _p_at(pred, e["peak_utc"])
        where = "" if math.isnan(e.get("az", float("nan"))) else f", {e['alt']:.0f}° up"
        rows.append(f"<li><b>{html.escape(e['name'])}</b> at {peak}{where}"
                    f"<br><span class='muted'>{html.escape(e['note'])}</span>"
                    f"<br><span class='p'>{_label(p, _high_at(pred, e['peak_utc']))}</span></li>")
    if scoreboard.get("n"):
        sb = (f"<p>Live record: {scoreboard['n']} hourly forecasts over {scoreboard['nights']} night(s). "
              f"Brier score (lower is better): model {scoreboard['brier_model']:.3f}, "
              f"raw forecast {scoreboard['brier_raw']:.3f}.</p>")
    else:
        sb = "<p>Live record: no hourly forecast has been scored yet. Forecasts are logged each run.</p>"
    s = offline["scores"]
    naive = s.get("naive threshold", {}).get("brier")
    off = (f"<p>Offline test on 2026 nights the model had not seen ({s[offline['best']]['n']} night hours with a "
           f"total-cloud observation): Brier {s[offline['best']]['brier']:.3f} for the model, "
           f"{s['raw forecast, calibrated']['brier']:.3f} for the calibrated raw forecast"
           + (f" and {naive:.3f} for the usual yes/no rule (clear if forecast cloud is under 25%)" if naive else "")
           + f". Gain over the calibrated raw forecast {offline['gain_vs_raw']:+.3f} "
           f"(95% interval {offline['gain_ci'][0]:+.3f} to {offline['gain_ci'][1]:+.3f}). That test used "
           f"reanalysis, not real forecasts, so the live record above is the number to trust.</p>")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Sky tonight, {html.escape(SITE['name'])}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;color:#111}}
li{{margin:.9rem 0;list-style:none}}ul{{padding:0}}.muted{{color:#555;font-size:.9rem}}.p{{font-weight:600}}
h2{{margin-top:2rem}}</style>
<h1>Sky tonight</h1><p>{html.escape(SITE['name'])}, {ev['day']}. Moon {ev['moon_illum']:.0%} lit.</p>
<ul>{''.join(rows)}</ul>
<h2>How far to trust the sky odds</h2>{sb}{off}
<p class="muted">The sky odds are for total cloud of at most 2 oktas (2/8) as reported in SYNOP observations at Barcelona
airport, which include high cloud. Those observations exist for only part of the hours, so the model rests on
them alone. Planet brightness is typical, not computed.
Meteor-shower dates are typical peaks. ISS passes use the latest public orbit and are not checked against
another predictor.</p></html>"""
