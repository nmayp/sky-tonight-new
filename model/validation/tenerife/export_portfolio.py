"""Write the portfolio files for the Tenerife check from the scored labels (run score_tenerife.py first).
Run from the repository root:
    python validation/tenerife/export_portfolio.py FRAMES_DIR PORTFOLIO_SKY_DIR
FRAMES_DIR is the frames folder made by make_frames.py (or validation/tenerife/frames).
Writes PORTFOLIO_SKY_DIR/cam.json, cam_meta.json and cam/<frame>.jpg (frames judged unreadable are not copied)."""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
frames, out = Path(sys.argv[1]), Path(sys.argv[2])
CREDIT = ("AllSkyKamera Network (https://allskykamera.space), camera ASK076 Deep Sky Teide, Tenerife "
          "(Joao Berbereia), https://allskykamera.space/kamera.php?k=ASK076")
scored = json.load(open(HERE / "labels_scored.json"))
labels = json.load(open(HERE / "labels.json"))
scores = json.load(open(HERE / "scores.json"))
(out / "cam").mkdir(parents=True, exist_ok=True)

rows = []
for o in scored:
    t = datetime.strptime(o["utc"], "%Y-%m-%dT%H:%M:%SZ")   # Tenerife civil time is UTC+0 in November
    rows.append({"file": f"cam/{o['file']}", "utc": o["utc"], "local": f"{t.day} {t:%b %H:%M}", "label": o["label"],
                 "p": round(o["p_model"], 2), "p_raw": round(o["p_raw"], 2), "cloud": round(o["cloud"]),
                 "lead": int(o["lead_h"]), "note": o["note"], "credit": CREDIT})
    shutil.copy(frames / o["file"], out / "cam" / o["file"])
json.dump(rows, open(out / "cam.json", "w"), indent=1, ensure_ascii=False)

m = scores["main"]
dropped = sum(1 for o in labels if o["label"] == "unreadable")
meta = {
    "camera": "AllSkyKamera ASK076 Deep Sky Teide",
    "place": "Tenerife, Canary Islands, Spain (camera at 275 m)",
    "lat": 28.2503, "lon": -16.8293,
    "credit": CREDIT,
    "licence_note": ("Shown under the Network's FAQ default ('use with attribution and link'). The Network's licence page "
                     "describes rights camera owners grant to the Network and does not itself license third parties, "
                     "so explicit permission from the Network is being requested separately."),
    "nights": m["nights"], "night_list": sorted({o["night"] for o in scored}),
    "frames_total": len(labels), "frames_dropped": dropped, "frames_used": m["n"],
    "brier_model": m["brier_model"], "brier_raw": m["brier_raw"], "brier_constant": m["brier_constant"],
    "hits": m["hits_model"], "sure": m["sure"], "hits_raw": m["hits_raw"],
    "hits_if_always_clear": sum(1 for o in scored if o["label"] == "clear"),
    "labels": {k: sum(1 for o in scored if o["label"] == k) for k in ("clear", "partly", "cloudy")},
    "sensitivity_brier": {k: scores["sensitivity_score_apo_rule"][k] for k in ("brier_model", "brier_raw", "brier_constant")},
    "method_note": (
        "One frame every two hours (20, 22, 00, 02, 04, 06 UT) from the camera's nightly timelapse videos, 13-21 Nov 2026, "
        "all with the Sun more than 12 degrees below the horizon. Each frame's time is the camera's own clock overlay "
        "(UT, checked against the Moon's rise and set), interpolated between readings on the first, middle and last frame of "
        "each video. Sky state (clear / partly / cloudy) is one reader's reading by eye following the project's rule; "
        "it is a judgement, not a measurement. Clear counts 1, partly 0.5, cloudy 0 in the Brier score. Forecasts are real "
        "NOAA GFS runs: the 12 UT run of the night's start date, 8-18 h ahead. 'hits' counts clear and cloudy frames only, "
        "with a hit when the model gave 50% or more (whole percent) and the frame was clear, or less than 50% and it was cloudy. "
        "The Moon was up for part of every night (18-92% lit over these nights), washes out faint stars and lights clouds from below; the camera's long "
        "exposures show more stars than an eye would, and drops on the dome blur some frames. Two frames were dropped as "
        "unreadable. Clouds and cloud seas at Tenerife differ from the mainland, and the model was fitted on Barcelona."),
}
json.dump(meta, open(out / "cam_meta.json", "w"), indent=1, ensure_ascii=False)
print(len(rows), "frames written to", out)
