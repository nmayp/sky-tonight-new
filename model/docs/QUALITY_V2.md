# Sky quality v2: what is predicted, from what, how well

v1 predicts one thing: the chance of a clear sky (total cloud at most 2 oktas). v2 predicts three things a person
looking up cares about, from more inputs, and is checked against v1 on the same table.

## What is predicted

| output | meaning | label (night-time SYNOP report) |
|---|---|---|
| `p` clear | chance of cloud at most 2 oktas | total cloud N <= 2 |
| `c` cloud | expected cloud cover in oktas (stored x10) | N, 0 to 8 |
| `h` poor transparency | chance of visibility under 10 km, or haze, dust, smoke, mist or fog | VV code under 10 km, or present weather ww 04-12, 28, 30-35, 40-49 |

Poor transparency includes rain and low cloud (they cut visibility too), so it is not dust alone. About 9.5% of test
reports are "poor".

A fourth model, sky brightness on the Stars4All TESS-W network, is described below; it is not in `forecast_v2.json`.

## From what

* Weather: NOAA GFS 0.5 degree forecasts as issued (12 UT cycle). v1's fields plus CAPE, model visibility, boundary-layer
  height, precipitable water, humidity at 850 and 700 hPa, and wind at 250 and 500 hPa with their shear.
* Aerosol: dust, aerosol optical depth and PM10 from the Open-Meteo air-quality API (CAMS). Its history is a
  best-estimate series, not an archived forecast, so the value at the forecast's valid time would be hindsight. The model uses the value 12 h
  before the cycle (what was known when the forecast was issued). The hindsight version is trained only to show how much it flatters (ablation).
* Site: latitude, longitude, station height. Terrain skyline and Moon are not used for cloud, clear or haze (Moon only in the TESS model).
* Lead time (`lead_h`) is an input, so the model fades toward what it knows as the forecast ages.

Why GFS from NOAA and not the Open-Meteo weather endpoints: the Previous Runs and Historical Forecast APIs answered
"daily request limit exceeded" (HTTP 429) for the build machine's address, and training on the same fields that are scored live keeps the two identical.
Open-Meteo is used for aerosol only. See `sky_tonight/quality/snapshot_v2.py`.

## Data

* Labels: Ogimet SYNOP, blocks 08 and 60, 2023-01-01 to 2026-11-20, hours 18, 21, 00, 03, 06 UT (about 700 requests,
  one at a time with a pause). 687 stations reported; 56 have enough labelled nights (>= 250 before 2026, >= 80 in 2026).
* Table: 417,229 rows (station, valid hour, lead), 56 stations, 2023-01 to 2026-10. Same-night leads 6, 9, 12, 15, 18 h every
  day; 36 to 228 h (valid 00 UT) every fourth day.
* Headline test: 2026-01 to 2026-10, leads 9, 12, 15 h, Sun below -6 degrees: 16,818 labelled rows (34 stations) for clear
  and cloud, 32,857 (51 stations) for poor transparency, 305 nights. v1's test had 8,462 rows from 32 stations.
* Training: through 2025-12. The saved model (`data/quality_v2.pkl`) is refit on everything through 2026-10, as v1's was.

## Results (test 2026-01 to 2026-10, 95% intervals resample whole nights)

| | climatology | plain forecast (calibrated) | v1-equivalent | v2 |
|---|---|---|---|---|
| clear, Brier | 0.2330 | 0.1710 | 0.1413 | 0.1382 (0.1326 to 0.1439) |
| poor transparency, Brier | 0.0764 | 0.0775 | 0.0639 | 0.0618 (0.0573 to 0.0666) |
| cloud, MAE in oktas | 2.41 | 1.83 | 1.47 | 1.45 (1.41 to 1.49) |

v2 minus v1-equivalent: clear -0.0030 Brier (interval -0.0042 to -0.0020), poor transparency -0.0021
(-0.0029 to -0.0014), cloud -0.026 oktas (-0.033 to -0.019). Small but outside the noise. Most of the lift over the plain forecast (about 17%
for clear) was already in v1; v2 adds about 2% on top.

Input groups (clear, Brier added to v1's inputs, cumulative): extra weather -0.0019, dust/aerosol -0.0004, upper winds -0.0002, site 0.
Dust at the valid time (hindsight) would add another -0.0011, more than the honest dust input is worth, so the CAMS history does flatter. For poor transparency
site (lat, lon) matters most (-0.0012), then extra weather. Full tables are in `skill_v2.json`.

Region hold-out (train without the region, test on it; Brier clear, v1-equivalent against v2): Catalonia (2 stations) 0.151 / 0.166 (v2 worse),
south 0.123 / 0.117, north 0.132 / 0.129, Canary Islands 0.223 / 0.204. Cloud oktas MAE: v2 worse only in Catalonia (1.63 against 1.52), better elsewhere. Poor transparency: v2 equal or better in all four.

Lead time: clear-sky Brier of v2 from 0.142 at 12 h to 0.222 at 228 h, against 0.233 for climatology; it stays below the plain forecast at every lead.

Camera check (Tenerife, 52 frames, 9 nights): v2 0.203, v1 0.389, calibrated plain forecast 0.239, constant guess 0.149.

Sky brightness (TESS-W): 91 Spanish photometers have at least 24 months of data (120 at least 12). 76 are within 35 km of a SYNOP station and have enough
readings. Target: how much brighter than the photometer's own dark baseline the sky is (mag). Error with Moon, calendar and the place's
baseline only: 0.53 mag; with weather: 0.38 mag; dust/aerosol and upper winds: no further gain. Yes/no "within 0.5 mag of its best": Brier 0.124 to 0.086.
AERONET: 53 Spanish sites listed, 27 have data in 2023-2026, 25 at least 12 months. Daytime sun photometers only, so not a night target; not modelled.

## Limits

See `limits` in `skill_v2.json`. Main points: labels are point observations at stations; poor transparency includes rain; dust history is a best-estimate series;
the Catalonia hold-out has 2 stations; station selection looked at how many labelled reports each station had, including 2026 (not values); Ogimet had a gap on 2025-08-15 and 29;
the camera check is small.

## Rebuild

Everything is in `sky_tonight/quality/`. From this directory (W = a work folder with about 1 GB free):

```
python -m sky_tonight.quality.synop_full --out $W/synop_a.csv --start 2023-01-01 --end 2023-12-31   # repeat per period, in parallel if you like
python -m sky_tonight.quality.gfs_table --stations data/quality/stations_spain_synop.json --out $W/gfs_table.csv
python -m sky_tonight.quality.gfs_table --stations data/quality/stations_spain_synop.json --out $W/gfs_table_b.csv --short-leads 6,18 --long-every 0
python -m sky_tonight.quality.build_table --synop "$W/synop_*.csv" --gfs $W/gfs_table.csv,$W/gfs_table_b.csv \
    --stations data/quality/stations_spain_synop.json --aq-cache $W/aq_cache.pkl --out $W/table.csv.gz
python -m sky_tonight.quality.train_quality --table $W/table.csv.gz --out data/quality_v2_scores.json --model-out data/quality_v2.pkl
python -m sky_tonight.quality.tess --counts tess_counts.json --out $W/tess_hourly.pkl --positions-out $W/tess_positions.json   # optional
python -m sky_tonight.quality.train_tess --tess $W/tess_hourly.pkl --positions $W/tess_positions.json --gfs $W/gfs_table.csv \
    --stations data/quality/stations_spain_synop.json --out data/quality_tess_scores.json
python -m sky_tonight.quality.export_public --scores data/quality_v2_scores.json --tess data/quality_tess_scores.json \
    --probes data/quality/sky_brightness_probes.json --tenerife validation/tenerife/scores_v2.json --out-dir <portfolio>/public/sky
```

Both download steps are resumable (`.done` files) and take a list of date ranges, so they can run as several processes. The GFS steps are limited by
network and Python overhead, the Ogimet step by the service's response time (a 4-day block takes 10 to 30 s); training takes about 7 minutes.

## Daily forecast

```
python -m sky_tonight.quality.snapshot_v2 --grid <portfolio>/public/sky/grid.json --model data/quality_v2.pkl --out <portfolio>/public/sky/forecast_v2.json
```

About 40 s. Uses the latest published 12 UT GFS run (10 days) and the Open-Meteo air-quality API; if that API fails the aerosol inputs are left empty
and the model still runs (`aerosol_inputs: "missing"`). `forecast_v2.json` has `p`, `c` (oktas x10), `h` (%) on the grid of `grid.json`, plus `dust` (ug/m3) and `aod` (x100) at the
valid hours. The sample workflow `.github-workflow-sky-refresh.yml` runs it daily at 17:50 UTC (not enabled).

Data credit: Ogimet (SYNOP), NOAA GFS (Open Data on AWS), Open-Meteo air-quality API (CAMS, https://open-meteo.com, non-commercial use, attribution),
Stars4All TESS-W network, AERONET (NASA).
