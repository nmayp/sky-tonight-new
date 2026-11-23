"""Regenerate the Tenerife validation frames from the AllSkyKamera ASK076 "Deep Sky Teide" nightly timelapse videos.

Run from the repository root:  python validation/tenerife/make_frames.py WORKDIR
It downloads the videos (about 9 MB each) into WORKDIR/vid, picks one frame every two hours, and writes
WORKDIR/frames/*.jpg (520 px wide, credit line burned in), WORKDIR/candidates.json and one contact sheet per
night (WORKDIR/sheets/) used for reading the sky by eye. Labels are not made here: they are written by hand
into validation/tenerife/labels.json.

Camera: AllSkyKamera Network (https://allskykamera.space), camera ASK076 Deep Sky Teide, Tenerife
(Joao Berbereia), https://allskykamera.space/kamera.php?k=ASK076

How each frame's true time is found: every video frame carries the camera's own clock in a text overlay
(dd-mm-yy - hh:mm:ss). The camera clock was checked to be UTC (the overlay's Moon rise/set, 15:12 / 04:04 on 20 Nov,
matches the computed 15:16 / 03:59 UT; Tenerife civil time is also UTC+0 in November). The overlay was read by eye
on the first, middle and last frame of each video (READINGS below, UT). Frames are evenly spaced (about 61.6 s
apart; checked on 13 frames per video for two videos, deviation under 30 s), so a frame's time is linear
interpolation between those readings. The contact sheets show the overlay of every chosen frame so the time can be
checked by eye.
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sky_tonight import events, get_site  # noqa: E402

CREDIT = ("AllSkyKamera Network (https://allskykamera.space), camera ASK076 Deep Sky Teide, Tenerife "
          "(Joao Berbereia), https://allskykamera.space/kamera.php?k=ASK076")
SHORT_CREDIT = "AllSkyKamera Network | ASK076 Deep Sky Teide (J. Berbereia)"
URL = "https://allskykamera.space/imageProxy.php?cam=ASK076&type=video&file=allsky-{d}.mp4"
NIGHTS = ["20261113", "20261114", "20261115", "20261116", "20261117", "20261118", "20261119", "20261120", "20261121"]
# Overlay clock (UT) on the first frame, the frame at index N//2, and the last frame (N = frame count, read with ffprobe).
READINGS = {
    "20261113": ("18:41:55", "00:52:44", "07:03:33"), "20261114": ("18:41:37", "00:53:27", "07:04:16"),
    "20261115": ("18:41:06", "00:52:54", "07:04:43"), "20261116": ("18:40:34", "00:53:25", "07:05:14"),
    "20261117": ("18:40:30", "00:53:20", "07:06:12"), "20261118": ("18:40:04", "00:53:57", "07:06:50"),
    "20261119": ("18:39:47", "00:53:36", "07:07:28"), "20261120": ("18:39:29", "00:54:21", "07:08:14"),
    "20261121": ("18:39:06", "00:53:56", "07:08:51"),
}
HOURS = [20, 22, 24, 26, 28, 30]          # UT hours after the start date's midnight (24 = 00 UT next day)
DISK = (300, 105, 1540, 1345)             # square crop of the sky disk in the 1920x1440 video frame


def hms(s):
    h, m, x = map(int, s.split(":"))
    return h * 3600 + m * 60 + x


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(out.stdout.strip())


def clock_to_seconds(night):
    """Seconds since 00 UT of the start date at the three read frames (the middle and last are after midnight)."""
    a, b, c = (hms(x) for x in READINGS[night])
    return a, b + 86400, c + 86400


def main(work):
    work = Path(work)
    for sub in ("vid", "raw", "frames", "sheets"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    site = get_site("deep_sky_teide")
    events.set_site(site)
    cands = []
    for night in NIGHTS:
        mp4 = work / "vid" / f"allsky-{night}.mp4"
        if not mp4.exists():
            r = requests.get(URL.format(d=night), timeout=300)
            if r.status_code != 200 or "video" not in r.headers.get("content-type", ""):
                print(night, "skipped:", r.status_code)
                continue
            mp4.write_bytes(r.content)
            time.sleep(3)
        N = frame_count(mp4)
        idx = [0, N // 2, N - 1]
        sec = clock_to_seconds(night)
        d0 = dt.datetime.strptime(night, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        picks = []
        for H in HOURS:
            n = int(round(np.interp(H * 3600, sec, idx)))
            t = d0 + dt.timedelta(seconds=float(np.interp(n, idx, sec)))
            picks.append((n, t))
        sel = "+".join(f"eq(n\\,{n})" for n, _ in picks)
        pat = work / "raw" / f"{night}_%02d.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", f"select='{sel}'", "-vsync", "0",
                        str(pat)], check=True)
        for k, (n, t) in enumerate(picks, 1):
            raw = Image.open(work / "raw" / f"{night}_{k:02d}.png")
            tm = t.replace(second=0, microsecond=0) + dt.timedelta(minutes=round(t.second / 60))
            stamp = f"{tm:%Y%m%dT%H%MZ}"
            disk = raw.crop(DISK).resize((520, 520), Image.LANCZOS).convert("RGB")
            out = Image.new("RGB", (520, 540), (0, 0, 0))
            out.paste(disk, (0, 0))
            ImageDraw.Draw(out).text((6, 524), SHORT_CREDIT, fill=(170, 170, 170))
            fn = f"cam_{stamp}.jpg"
            out.save(work / "frames" / fn, quality=80, optimize=True, comment=CREDIT.encode())
            ov = raw.crop((715, 0, 1135, 60))
            alt, illum = events.moon_info([tm])
            sun = float(events.sun_alt([tm])[0])
            cands.append({"file": fn, "utc": tm.strftime("%Y-%m-%dT%H:%M:00Z"), "night": f"{tm - dt.timedelta(hours=12):%Y-%m-%d}",
                          "video": night, "frame_index": n, "sun_alt": round(sun, 1),
                          "moon_alt": round(float(alt[0]), 0), "moon_illum": round(float(illum[0]), 2)})
            ov.save(work / "raw" / f"ov_{fn[:-4]}.png")
    # contact sheets (full-resolution disk, 3 x 2 per night) with the overlay clock under every thumbnail
    for night in NIGHTS:
        cs = [c for c in cands if c["video"] == night]
        if not cs:
            continue
        sheet = Image.new("RGB", (3 * 520, 2 * 590), (20, 20, 20))
        for i, c in enumerate(cs):
            im = Image.open(work / "frames" / c["file"]).crop((0, 0, 520, 520))
            sheet.paste(im, ((i % 3) * 520, (i // 3) * 590))
            ov = Image.open(work / "raw" / f"ov_{c['file'][:-4]}.png").resize((280, 40))
            sheet.paste(ov, ((i % 3) * 520 + 20, (i // 3) * 590 + 522))
            ImageDraw.Draw(sheet).text(((i % 3) * 520 + 320, (i // 3) * 590 + 530),
                                       f"#{i + 1}  {c['utc'][5:16]}Z", fill=(255, 255, 0))
        sheet.save(work / "sheets" / f"sheet_{night}.png")
    (work / "candidates.json").write_text(json.dumps(cands, indent=1))
    print(len(cands), "frames")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tenerife_work")
