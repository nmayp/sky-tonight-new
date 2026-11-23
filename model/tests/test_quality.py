"""Small tests for the v2 sky-quality parsing and feature code (no network)."""
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sky_tonight.quality import aq, evaluate, features as F, gfs_table as G, models, snapshot_v2, synop_full as S

LINE_CLEAR = "08181,2026,03,01,00,00,AAXX 01004 08181 02480 73303 10127 20099 30252 40259 58004 60001 8655/ 333 60007=="
LINE_HAZE = "08004,2026,03,01,00,00,AAXX 01001 08004 07/36 /2205 10093 20071 30152 40263 5/011 60001 710// 333 55300="
LINE_WX = "08001,2026,03,01,03,00,AAXX 01034 08001 41070 30905 10132 20098 30078 40158 58003 69961 72366 333 69967="


def test_visibility_codes():
    assert S.vv_km(80) == 30.0 and S.vv_km(60) == 10.0 and S.vv_km(59) == 9.0
    assert S.vv_km(10) == 1.0 and S.vv_km(97) == 10.0 and S.vv_km(99) == 50.0 and S.vv_km(53) is None


def test_parse_clear_line_with_cloud_and_visibility():
    r = S.parse_line(LINE_CLEAR)
    assert r["wmo"] == "08181" and r["utc"] == "2026-03-01T00:00:00Z"
    assert r["N"] == 7 and r["vv_km"] == 30.0
    assert r["ww"] == 0 and r["wx_obs"] == 1          # iX = 2: nothing significant


def test_parse_automatic_station_without_cloud_but_with_weather_group():
    r = S.parse_line(LINE_HAZE)
    assert r["N"] is None and r["vv_km"] == 3.6 and r["ww"] == 10
    assert S.poor_transparency(r["vv_km"], r["ww"], r["wx_obs"]) == 1.0


def test_parse_weather_group_when_included():
    r = S.parse_line(LINE_WX)
    assert r["ww"] == 23 and r["N"] == 3
    # the 7-group is read, not a 7 elsewhere: precipitation group 6RRRt must not be taken for it
    assert S.parse_line("x,2026,03,01,03,00,AAXX 01034 08001 21070 30905 10132 20098 6//// 333=")["ww"] is None


def test_parse_rejects_nil_and_other_hours_filtered():
    assert S.parse_line("08001,2026,03,01,03,00,AAXX 01034 08001 NIL=") is None
    rows = S.parse_text(LINE_CLEAR + "\n" + LINE_CLEAR.replace(",00,00,AAXX", ",10,00,AAXX"))
    assert len(rows) == 1 and rows[0]["utc"].endswith("T00:00:00Z")


def test_poor_transparency_rules():
    assert S.poor_transparency(9.0, 0, 1) == 1.0          # visibility under 10 km
    assert S.poor_transparency(25.0, 5, 1) == 1.0         # haze reported
    assert S.poor_transparency(25.0, 0, 1) == 0.0
    assert S.poor_transparency(10.0, None, 0) == 0.0      # visibility alone is enough to say good
    assert math.isnan(S.poor_transparency(None, None, 0))


def test_sun_altitude_matches_known_values():
    # Madrid 21 June 2026 at 12:15 UT: Sun near its highest; same place at 00 UT: well below the horizon
    hi = F.sun_altitude(pd.DatetimeIndex(["2026-06-21T12:15:00Z"]), 40.4, -3.7)[0]
    lo = F.sun_altitude(pd.DatetimeIndex(["2026-06-21T00:00:00Z"]), 40.4, -3.7)[0]
    assert 65 < hi < 75 and lo < -20


def test_feature_groups_nest_and_hindsight_swap():
    assert set(F.V1_COLS) < set(F.v2_cols()) and "lead_h" in F.v2_cols(lead=True)
    h = F.cols_for("+ site (lat, lon)", hindsight_aerosol=True)
    assert "dust_v" in h and "dust_0" not in h


def test_index_parsing_and_wanted_ranges():
    idx = G.parse_index("10:100:d=2026112112:TCDC:entire atmosphere:12 hour fcst:\n"
                        "11:200:d=2026112112:TCDC:entire atmosphere:6-12 hour ave fcst:\n"
                        "12:300:d=2026112112:CAPE:surface:12 hour fcst:\n")
    r = G.wanted_ranges(idx, 12)
    assert r["cloud_cover"] == (100, 199) and r["cape"] == (300, None) and "cloud_cover_low" not in r


def test_point_weights_sum_to_one_and_hit_grid_node():
    ps = G.PointSet({"a": (40.0, -3.0), "b": (40.2, -3.1)})
    ps._build(720, 361, 90.0, 0.5, 0.5, False)
    flat, w, _ = ps._geom
    assert np.allclose(w.sum(1), 1.0) and flat.shape == (2, 4)
    # a point exactly on a node is dominated by that node (distance floor 0.5 km)
    assert w[0].max() > 0.9


def test_derive_units():
    raw = {k: np.array([1.0]) for k in G.RAW}
    raw.update(t2m_k=np.array([283.15]), d2m_k=np.array([283.15]), u10=np.array([3.0]), v10=np.array([4.0]),
               sp_pa=np.array([101300.0]), prate=np.array([1e-4]))
    d = G.derive(raw)
    assert abs(d["relative_humidity_2m"][0] - 100) < 1e-6 and abs(d["wind_speed_10m"][0] - 18.0) < 1e-9
    assert abs(d["surface_pressure"][0] - 1013) < 1e-9 and abs(d["precipitation"][0] - 0.36) < 1e-9


def test_job_list_structure():
    j = G.job_list("2023-01-01", "2023-01-08")
    assert {h for _, h in j} >= {9, 12, 15, 36, 228}
    assert sum(1 for _, h in j if h == 9) == 8 and sum(1 for _, h in j if h == 36) == 2
    assert all(h in (6, 18) for _, h in G.job_list("2023-01-01", "2023-01-03", long_every=0, short_leads=(6, 18)))


def test_air_quality_parse_single_and_batch():
    one = {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "dust": [3.0, 4.0], "aerosol_optical_depth": [None, 0.1], "pm10": [10.0, 11.0]}}
    df = aq.parse_response(one)[0]
    assert df.index.tz is not None and np.isnan(df.aerosol_optical_depth.iloc[0]) and df.dust.iloc[1] == 4.0
    assert len(aq.parse_response([one, one])) == 2


def test_valid_times_cover_ten_days_in_three_hour_steps():
    cyc = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
    v = snapshot_v2.valid_times(cyc)
    assert len(v) == 50 and v[0][0] == 6 and v[-1][0] == 234 and all(l % 3 == 0 for l, _ in v)
    assert [t.hour for _, t in v[:5]] == [18, 21, 0, 3, 6]


def test_bootstrap_paired_difference_and_night_resampling():
    nights = np.repeat(np.arange(50), 10)
    a, b = np.full(500, 0.1), np.full(500, 0.2)
    r = evaluate.bootstrap(nights, {"a": a, "b": b}, [("a", "b")], reps=200)
    assert abs(r["diff"]["a-b"]["point"] + 0.1) < 1e-12 and r["diff"]["a-b"]["ci"][1] < 0 and r["n_nights"] == 50


def test_climatology_station_month_with_fallback():
    t = pd.to_datetime(["2025-01-05", "2025-01-06", "2025-02-01"], utc=True)
    X = pd.DataFrame({"wmo": ["a", "a", "b"], "t": t})
    c = models.Clim("station_month", min_n=2).fit(X, [1.0, 0.0, 1.0])
    p = c.predict(pd.DataFrame({"wmo": ["a", "b"], "t": pd.to_datetime(["2026-01-10", "2026-02-10"], utc=True)}))
    assert p[0] == 0.5 and p[1] == 1.0      # station-month where enough data, month mean otherwise
