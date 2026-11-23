"""Write skill_v2.json and quality_v2_meta.json for the site from the saved score files.

    python -m sky_tonight.quality.export_public --scores data/quality_v2_scores.json --tess data/quality_tess_scores.json \
        --probes data/quality/sky_brightness_probes.json --out-dir ../../../portfolio/public/sky
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

V1, V2, RAW, CLIM = "v1-equivalent", "v2", "plain forecast (calibrated)", "climatology (station x month)"

SOURCES = [
    "Station labels: SYNOP reports from Ogimet (https://www.ogimet.com), WMO blocks 08 and 60, synoptic hours 18-06 UT: total cloud (oktas), visibility (VV), present weather (ww).",
    "Weather inputs: NOAA GFS 0.5 degree forecasts as issued (12 UT cycle, leads 6 to 228 h), NOAA Open Data on AWS (noaa-gfs-bdp-pds), read by byte range.",
    "Aerosol inputs: dust, aerosol optical depth, PM10 from the Open-Meteo air-quality API (CAMS model fields; https://open-meteo.com, non-commercial use, attribution required).",
    "Sky brightness: Stars4All TESS-W photometer network (https://tess.dashboards.stars4all.eu), 76 Spanish photometers.",
]
LIMITS = [
    "Labels are airport and synoptic-station SYNOP reports, a point observation by an observer or an automatic sensor, not the whole sky seen from a dark site.",
    "Poor transparency = visibility under 10 km or haze, dust, smoke, mist or fog reported. It includes rain and low cloud, so it is not dust alone.",
    "Dust and aerosol history from the API is a CAMS best-estimate, not an archived forecast; the model uses the value 12 h before the forecast cycle, and the value at the valid time (which would flatter it) is shown separately in the ablation.",
    "Weather inputs come from NOAA GFS (0.5 degree), not the Open-Meteo weather endpoints: those were rate-limited for the sandbox address on the day of the build, and using one source for training and live keeps the inputs identical.",
    "Hyper-parameters were chosen on 2025 only; the 2026 test period was never used to choose anything. Station selection used how many labelled night reports each station had, including in 2026 (not their values).",
    "Camera check (9 Tenerife nights, 52 frames, one reader, nothing tuned): v2 Brier 0.203, v1 0.389, calibrated plain forecast 0.239, constant guess 0.149. Better than v1 and than the plain forecast, still not better than a constant guess, and v2 says 56% clear where the frames read 75%.",
    "Ogimet returned no data for the mainland block on 2025-08-15 and little on 2025-08-29 (source gap, not retried further).",
    "Region hold-out for Catalonia rests on 2 stations (clear and cloud) and is not reliable.",
]


def pct(a, b):
    return round(100.0 * (a - b) / b, 1)


def build(scores, tess, probes, tenerife=None):
    c, h, k = scores["clear"], scores["haze"], scores["oktas"]
    hc, hh, hk = c["headline"], h["headline"], k["headline"]

    def row(hl, key):
        m = hl["models"]
        ci = hl.get("ci_brier") or hl.get("ci_mae")
        return {"n": hl["n"], "stations": hl["stations"], "nights": hl["nights"], "base_rate": m[V2].get("base_rate"),
                "climatology": m[CLIM][key], "climatology_month": m["climatology (month, all stations)"][key], "plain_forecast": m[RAW][key],
                "v1_equivalent": m[V1][key], "v2": m[V2][key],
                "ci": {"climatology": ci[CLIM], "plain_forecast": ci[RAW], "v1_equivalent": ci[V1], "v2": ci[V2]},
                "v2_minus_v1": hl["diff"][f"{V2}-{V1}"], "v2_minus_plain_forecast": hl["diff"][f"{V2}-{RAW}"],
                "v2_minus_climatology": hl["diff"][f"{V2}-{CLIM}"]}
    clear, haze = row(hc, "brier"), row(hh, "brier")
    cloud = row(hk, "mae")
    for r, hl in ((clear, hc), (haze, hh)):
        r["v2_auc"], r["v1_auc"], r["raw_auc"] = hl["models"][V2]["auc"], hl["models"][V1]["auc"], hl["models"][RAW]["auc"]
        r["v2_ece"], r["v1_ece"], r["raw_ece"] = hl["models"][V2]["ece"], hl["models"][V1]["ece"], hl["models"][RAW]["ece"]
        r["v2_logloss"], r["v1_logloss"], r["raw_logloss"], r["clim_logloss"] = (hl["models"][V2]["logloss"], hl["models"][V1]["logloss"],
                                                                                  hl["models"][RAW]["logloss"], hl["models"][CLIM]["logloss"])
        r["logloss_ci"] = hl["ci_logloss"]
        r["logloss_v2_minus_v1"] = hl["diff_logloss"][f"{V2}-{V1}"]
    cloud["rmse"] = {k_: hk["models"][k_]["rmse"] for k_ in (CLIM, RAW, V1, V2)}
    cloud["bias_v2"] = hk["models"][V2]["bias"]
    cloud["climatology_median_mae"] = hk["models"]["climatology median (station x month)"]["mae"]
    cloud["plain_forecast_uncalibrated_mae"] = hk["models"]["plain forecast (uncalibrated: cloud % / 100 x 8)"]["mae"]

    def lead_rows(t, key):
        out = []
        for r in scores[t]["lead_curve"]["rows"]:
            out.append({"lead_h": r["lead_h"], "n": r["n"], "nights": r["nights"], "model": round(r["v2"][key], 4), "v1_equivalent": round(r["v1"][key], 4),
                        "raw": round(r["raw"][key], 4), "clim": round(r["clim"][key], 4), "best_weight": r.get("best_weight"),
                        "blend": round(r["blend"][key], 4) if "blend" in r else None, **({"auc": round(r["v2"]["auc"], 3)} if "auc" in r["v2"] else {})})
        return out

    better = clear["v2_minus_v1"]["ci"][1] < 0
    verdict = (
        "Use v2, mainly for what it adds. For clear sky v2 is a small but real step over v1-equivalent "
        f"(Brier {clear['v2']:.4f} against {clear['v1_equivalent']:.4f}, a gain of {abs(clear['v2_minus_v1']['point']):.4f} or "
        f"{abs(pct(clear['v2'], clear['v1_equivalent'])):.1f}%; 95% interval of the difference {clear['v2_minus_v1']['ci'][0]:.4f} to {clear['v2_minus_v1']['ci'][1]:.4f}); "
        f"both are about {abs(pct(clear['v2'], clear['plain_forecast'])):.0f}% better than the calibrated plain forecast, so most of the lift was already in v1. "
        f"The new outputs are useful: expected cloud cover has MAE {cloud['v2']:.2f} oktas against {cloud['plain_forecast']:.2f} for the calibrated plain forecast "
        f"({abs(pct(cloud['v2'], cloud['plain_forecast'])):.0f}% lower) and poor transparency has Brier {haze['v2']:.4f} against {haze['plain_forecast']:.4f} "
        f"({abs(pct(haze['v2'], haze['plain_forecast'])):.0f}% lower) and {haze['climatology']:.4f} for station-month climatology. "
        "Extra weather fields (CAPE, boundary-layer height, model visibility, precipitable water, humidity at 850 and 700 hPa) give most of the gain; "
        "upper winds add little; dust and aerosol add almost nothing once hindsight is removed. "
        "Caveats: v2 is worse than v1-equivalent in the Catalonia hold-out (2 stations); on the 9-night Tenerife camera check v2 beats v1 and the plain forecast but not a constant guess (52 frames, one reader). "
        "If only P(clear) is needed, v1 and v2 are practically the same; keep v1 for it only if keeping the published 2026 numbers unchanged matters more than the three outputs.")
    skill = {
        "version": "v2", "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "train_through": "2025-12-31", "test_from": "2026-01-01", "test_to": "2026-10-31", "final_model_trained_through": "2026-10-31",
        "protocol": {
            "split": "train 2023-01..2025-12, test 2026-01..2026-10, nights only (Sun below -6 degrees at the station)",
            "headline_rows": "same-night leads 9, 12 and 15 h (21, 00 and 03 UT valid times from the 12 UT cycle); leads 6 and 18 h (dusk and dawn) were used in training and are scored separately",
            "baselines": "climatology = mean of the target by station and calendar month from the training years (month only, all stations, also shown); plain forecast = isotonic map from GFS total cloud (clear, cloud oktas) or GFS model visibility (poor transparency) fitted on the training years; v1-equivalent = v1's inputs, model and calibration recipe retrained on this table",
            "intervals": "95% intervals resample whole nights (2000 resamples for the headline, 1000 for ablation and hold-out); paired differences use the same resamples",
            "tuning": "four boosting settings compared on 2025 with models trained on 2023-2024, separately for v1-equivalent and v2; test year not used",
            "deviations": [
                "Weather inputs are NOAA GFS 0.5 degree as issued (12 UT cycle), not the Open-Meteo Previous Runs or Historical Forecast APIs: those answered 'daily limit exceeded' (HTTP 429, resets in about 11 h) for the sandbox address, so they could not be used.",
                "Lead structure: leads 6, 9, 12, 15, 18 h for every day, and 36 to 228 h (valid 00 UT) every fourth day, all from the 12 UT cycle.",
                "More stations than v1: 56 Spanish SYNOP stations kept (34 with cloud labels in the headline test, 51 with visibility), against 32 in v1; the v1-equivalent model is retrained on the same table.",
                "Aerosol inputs are the CAMS best-estimate 12 h before the cycle, not the value at the valid time (see ablation row with hindsight).",
                "No terrain skyline input: it describes a site's horizon, not the station's cloud or haze, and v1's horizon script is for places, not these stations.",
                "Moon features are used only in the sky-brightness (TESS) model, never for cloud or haze.",
            ]},
        "data": {"sources": SOURCES, "table_rows": scores["table"]["rows"], "stations_in_table": scores["table"]["stations"],
                 "table_first_valid_utc": scores["table"]["first"], "table_last_valid_utc": scores["table"]["last"],
                 "train_rows_clear": hc["train_rows"], "train_rows_cloud": hk["train_rows"], "train_rows_haze": hh["train_rows"],
                 "test_rows": {"clear": hc["n"], "cloud": hk["n"], "poor_transparency": hh["n"]},
                 "test_stations": {"clear": hc["stations"], "cloud": hk["stations"], "poor_transparency": hh["stations"]}, "test_nights": hc["nights"]},
        "clear": {**clear, "reliability": hc["reliability"], "ablation": c["ablation"], "region_holdout": c["region_holdout"], "lead": lead_rows("clear", "brier"),
                  "dusk_dawn": hc.get("dusk_dawn"), "hyperparameters": c["params"]},
        "poor_transparency": {**haze, "definition": "visibility under 10 km, or haze, dust, smoke, mist or fog in present weather (SYNOP VV and ww)",
                              "reliability": hh["reliability"], "ablation": h["ablation"], "region_holdout": h["region_holdout"], "lead": lead_rows("haze", "brier"),
                              "dusk_dawn": hh.get("dusk_dawn"), "hyperparameters": h["params"]},
        "cloud_oktas": {**cloud, "reliability": hk["reliability"], "ablation": k["ablation"], "region_holdout": k["region_holdout"], "lead": lead_rows("oktas", "mae"),
                        "dusk_dawn": hk.get("dusk_dawn"), "hyperparameters": k["params"]},
        "sky_brightness": {"probe": probes, "model": tess, "built": True,
                           "summary": "TESS-W: a model of how much brighter than its own dark baseline a place's sky is, with Moon and calendar alone as the baseline; weather inputs cut the error a lot, dust and aerosol add nothing measurable."},
        "tenerife_camera_check": tenerife,
        "verdict": verdict, "limits": LIMITS,
    }
    meta = {
        "verdict": ("Keep v2 for all three outputs. Clear sky: small gain over v1 (Brier %.4f vs %.4f). New: expected cloud (MAE %.2f oktas vs %.2f plain forecast) and poor transparency (Brier %.4f vs %.4f). "
                    "Most of the lift over the plain forecast was already in v1; dust and aerosol add almost nothing.") % (
            clear["v2"], clear["v1_equivalent"], cloud["v2"], cloud["plain_forecast"], haze["v2"], haze["plain_forecast"]),
        "verdict_short": "v2 for all three; clear-sky gain over v1 is small",
        "clear": {"v1_brier": clear["v1_equivalent"], "v2_brier": clear["v2"], "raw_brier": clear["plain_forecast"], "clim_brier": clear["climatology"], "ci": clear["ci"],
                  "v2_minus_v1_ci": clear["v2_minus_v1"]["ci"], "v2_auc": clear["v2_auc"], "n": clear["n"]},
        "haze": {"v1_brier": haze["v1_equivalent"], "v2_brier": haze["v2"], "raw_brier": haze["plain_forecast"], "clim_brier": haze["climatology"], "ci": haze["ci"],
                 "v2_minus_v1_ci": haze["v2_minus_v1"]["ci"], "v2_auc": haze["v2_auc"], "base_rate": haze["base_rate"], "n": haze["n"]},
        "cloud": {"v1_mae": cloud["v1_equivalent"], "v2_mae": cloud["v2"], "raw_mae": cloud["plain_forecast"], "clim_mae": cloud["climatology"], "ci": cloud["ci"], "n": cloud["n"]},
        "n_test": {"clear": clear["n"], "cloud": cloud["n"], "haze": haze["n"]},
        "stations": {"clear_test": clear["stations"], "haze_test": haze["stations"], "in_table": scores["table"]["stations"]},
        "trained_through": "2025-12-31 (final model: 2026-10-31)", "test_period": "2026-01-01 to 2026-10-31",
        "sky_brightness": {"tess_photometers_spain_with_24_months": probes["tess_w"]["ge_24_months"], "model_photometers": tess["photometers"],
                           "mae_mag_moon_baseline": tess["dimming"]["moon + calendar + place baseline"]["mae"],
                           "mae_mag_with_weather": tess["dimming"]["+ upper winds"]["mae"], "aeronet_sites_ge_12_months": probes["aeronet"]["sites_ge_12_months"]},
        "sources": SOURCES, "limits": LIMITS,
    }
    return skill, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tess", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--tenerife", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ten = json.load(open(a.tenerife)) if a.tenerife and Path(a.tenerife).exists() else None
    skill, meta = build(json.load(open(a.scores)), json.load(open(a.tess)), json.load(open(a.probes)), ten)
    out = Path(a.out_dir)
    json.dump(skill, open(out / "skill_v2.json", "w"), indent=1, default=float)
    json.dump(meta, open(out / "quality_v2_meta.json", "w"), indent=1, default=float)
    print("wrote", out / "skill_v2.json", out / "quality_v2_meta.json")


if __name__ == "__main__":
    main()
