import datetime as dt
import io

import numpy as np
import pandas as pd

from sky_tonight import events, labels, model


def test_cavok_counts_as_clear(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("station,valid,skyc1,skyc2,skyc3,skyc4,skyl1,wxcodes\n"
                   "LEBL,2026-01-01 00:00,M,M,M,M,M,M\n"
                   "LEBL,2026-01-01 01:00,FEW,M,M,M,3000,M\n"
                   "LEBL,2026-01-01 02:00,SCT,BKN,M,M,3000,M\n")
    d = labels.load_metar(str(csv))
    assert list(d.clear) == [1, 1, 0]
    assert d.octas.iloc[2] == 6.0  # worst layer wins


def test_nearest_report_kept_per_hour(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("station,valid,skyc1,skyc2,skyc3,skyc4,skyl1,wxcodes\n"
                   "LEBL,2026-01-01 00:50,OVC,M,M,M,M,M\n"
                   "LEBL,2026-01-01 01:05,FEW,M,M,M,M,M\n")
    d = labels.load_metar(str(csv))
    # both reports belong to the 01:00 hour; the one 5 minutes away beats the one 10 minutes away
    assert len(d) == 1 and d.clear.iloc[0] == 1


def test_ece_is_zero_for_calibrated_forecast():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1] * 50)
    p = np.full(len(y), 0.7)
    assert model.ece(y, p) < 1e-9


def test_scores_reward_sharper_correct_forecasts():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    good = model.score(y, p)
    flat = model.score(y, np.full(5000, y.mean()))
    assert good["brier"] < flat["brier"] and good["auc"] > 0.7


def test_moon_illumination_and_night_window():
    times = events.night_window(dt.date(2026, 11, 21))
    alt, illum = events.moon_info(times)
    assert len(times) == 14 and 0.8 < illum.max() < 1.0  # about 3 days before full moon
    assert events.sun_alt(times).min() < -30  # mid-winter night is properly dark


def test_site_renders_without_forecast():
    from sky_tonight import site
    ev = {"day": "2026-11-21", "moon_illum": 0.5,
          "events": [{"kind": "note", "name": "x", "note": "y"}]}
    pred = pd.DataFrame({"p_model": [], "cloud_high": []}, index=pd.DatetimeIndex([], tz="UTC"))
    offline = {"scores": {"a": {"brier": 0.1, "n": 10}, "raw forecast, calibrated": {"brier": 0.2}}, "best": "a",
               "gain_vs_raw": 0.1, "gain_ci": [0.05, 0.15]}
    assert "Sky tonight" in site.render(ev, pred, {"n": 0}, offline)


def test_synop_parse_reads_okta_and_skips_unobserved():
    from sky_tonight import synop
    txt = ("08181,2026,11,20,15,00,AAXX 20154 08181 22/80 62404 10183=\n"
           "08181,2026,11,20,16,00,AAXX 20164 08181 45/80 /2403 10173=\n"
           "08181,2026,11,20,17,00,AAXX 2017/ 08181 NIL=\n")
    d = synop.parse(txt)
    assert list(d.N.fillna(-1)) == [6.0, -1.0, -1.0]


def test_naive_threshold_is_worse_than_calibrated_when_cloud_is_informative():
    rng = np.random.default_rng(3)
    cc = rng.uniform(0, 100, 4000)
    y = (rng.uniform(size=4000) < 1 - cc / 100).astype(int)
    X = pd.DataFrame({"cloud_cover": cc}, index=pd.date_range("2025-01-01", periods=4000, freq="h", tz="UTC"))
    naive = model.NaiveThreshold().fit(X, y).predict(X)
    cal = model.RawCloudBaseline().fit(X, y).predict(X)
    assert model.score(y, cal)["brier"] < model.score(y, naive)["brier"]


def test_gfs_unit_conversion_to_model_features():
    from sky_tonight import gfs
    v = {"cloud_cover": 50, "cloud_cover_low": 10, "cloud_cover_mid": 20, "cloud_cover_high": 30,
         "t2m_k": 288.15, "d2m_k": 278.15, "u10": 3.0, "v10": 4.0, "sp_pa": 101325.0, "prate": 0.001}
    f = gfs.to_model_features(v)
    assert abs(f["temperature_2m"] - 15.0) < 1e-6 and abs(f["dew_point_2m"] - 5.0) < 1e-6
    assert abs(f["wind_speed_10m"] - 18.0) < 1e-6      # 5 m/s -> 18 km/h
    assert abs(f["surface_pressure"] - 1013.25) < 1e-6
    assert abs(f["precipitation"] - 3.6) < 1e-6        # 0.001 kg m-2 s-1 -> 3.6 mm/h
    assert 40 < f["relative_humidity_2m"] < 60         # 15 C with a 5 C dew point is about 47%


def test_gfs_url_and_cycle_layout():
    from datetime import datetime, timezone
    from sky_tonight import gfs
    u = gfs._url(datetime(2026, 11, 17, 18, tzinfo=timezone.utc), 12, "0p25")
    assert u.endswith("gfs.20261117/18/atmos/gfs.t18z.pgrb2.0p25.f012")


def test_site_features_adds_the_columns_the_multi_model_expects():
    import pandas as pd
    from sky_tonight import live_gfs
    idx = pd.DatetimeIndex(["2026-12-01 21:00", "2026-12-02 00:00"], tz="UTC")
    df = pd.DataFrame({"temperature_2m": [10.0, 8.0], "dew_point_2m": [4.0, 6.0]}, index=idx)
    x = live_gfs.site_features(df, 2000)
    assert list(x.dew_depression) == [6.0, 2.0] and x.elev_km.iloc[0] == 2.0
    assert {"month_sin", "month_cos", "hour_sin", "hour_cos"} <= set(x.columns)


def test_region_labels_put_barcelona_in_catalonia_and_tenerife_in_the_canaries():
    import pandas as pd
    from sky_tonight import train_multi
    d = pd.DataFrame({"wmo": ["08181", "60025", "08221"], "lat": [41.28, 28.05, 40.45], "lon": [2.07, -16.57, -3.55]})
    assert list(train_multi.region(d)) == ["catalonia", "canary", "interior/other"]


def test_naive_and_raw_models_in_train_multi_behave():
    import numpy as np
    import pandas as pd
    from sky_tonight import train_multi as tm
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"cloud_cover": rng.uniform(0, 100, 3000)})
    y = (rng.uniform(size=3000) < 1 - X.cloud_cover / 100).astype(int)
    raw = tm.Raw().fit(X, y).predict(X)
    naive = tm.Naive().fit(X, y).predict(X)
    assert tm.score(y, raw)["brier"] < tm.score(y, naive)["brier"]
