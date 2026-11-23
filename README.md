# Sky Tonight

For any place in Spain and any night up to about six months ahead: the chance of a clear sky, haze risk, and cloud
cover, plus what you can see (Moon, planets, stars, satellites), dimmed by city light. A gradient-boosted model turns
NOAA GFS forecasts into calibrated outputs; past the forecast range the page falls back to the monthly average.

- `model/` Python: training, validation, tests. Run `pip install -r requirements.txt && pytest`, then `python -m sky_tonight planner`.
- `web/` Next.js page. Run `npm install && npm run dev` (or `npm run build`). It reads `web/public/sky/*.json` in the browser.

Rebuilding the json: `forecast_v2.json` with `python -m sky_tonight.quality.snapshot_v2 --grid grid.json --model data/quality_v2.pkl --out forecast_v2.json`;
`grid.json` and `skill.json` with `python -m sky_tonight.quality.export_public`; see `model/docs/QUALITY_V2.md` for the full rebuild sequence.

Measured on 2026 nights the model never saw (56 stations, 16,818 labelled rows, leads 9–15 h):

| | Brier (lower is better) |
|---|---|
| climatology | 0.233 |
| plain forecast, calibrated | 0.171 |
| gradient boosting v2 | 0.138 (0.133–0.144) |

Camera check (Tenerife all-sky, 52 frames, 9 nights): v2 Brier 0.203 vs. plain forecast 0.239 vs. constant 0.149.
Catalonia hold-out (2 stations): v2 worse than v1. "Clear" is a station observer's estimate (0–2 oktas).
