"""Models and baselines for the v2 sky-quality targets (kept in a module so pickles load by name)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

# v1's settings, then three alternatives; the choice among them is made on 2025 (never on the test year)
PARAM_GRID = [
    dict(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=60, l2_regularization=1.0),
    dict(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=60, l2_regularization=1.0),
    dict(max_depth=5, learning_rate=0.05, max_iter=400, min_samples_leaf=100, l2_regularization=1.0),
    dict(max_depth=6, learning_rate=0.03, max_iter=500, min_samples_leaf=150, l2_regularization=2.0),
]


def _time_split(X: pd.DataFrame, y, frac: float):
    order = np.argsort(X["t"].values, kind="stable")      # calibrate on the latest part of the training data
    X, y = X.iloc[order], np.asarray(y)[order]
    cut = int(len(X) * (1 - frac))
    return X, y, cut


class Boost:
    """Gradient boosted classifier plus isotonic calibration on the latest quarter of the training rows (v1 recipe)."""

    def __init__(self, cols, params=None, calib_frac=0.25):
        self.cols, self.params, self.calib_frac = list(cols), dict(params or PARAM_GRID[0]), calib_frac

    def fit(self, X, y):
        X, y, cut = _time_split(X, y, self.calib_frac)
        self.m = HistGradientBoostingClassifier(random_state=0, **self.params).fit(X[self.cols].iloc[:cut], y[:cut])
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            self.m.predict_proba(X[self.cols].iloc[cut:])[:, 1], y[cut:])
        return self

    def predict(self, X):
        return self.iso.predict(self.m.predict_proba(X[self.cols])[:, 1])


class BoostReg:
    """Expected cloud cover in oktas: boosted regressor (squared error, so it estimates the mean) plus isotonic
    calibration of the mean on the latest quarter of the training rows."""

    def __init__(self, cols, params=None, calib_frac=0.25):
        self.cols, self.params, self.calib_frac = list(cols), dict(params or PARAM_GRID[0]), calib_frac

    def fit(self, X, y):
        X, y, cut = _time_split(X, y, self.calib_frac)
        self.m = HistGradientBoostingRegressor(random_state=0, **self.params).fit(X[self.cols].iloc[:cut], y[:cut])
        self.iso = IsotonicRegression(y_min=0, y_max=8, out_of_bounds="clip").fit(self.m.predict(X[self.cols].iloc[cut:]), y[cut:])
        return self

    def predict(self, X):
        return np.clip(self.iso.predict(self.m.predict(X[self.cols])), 0, 8)


class RawIso:
    """Plain forecast, calibrated: monotone map from one forecast field to the target (v1's 'raw calibrated')."""

    def __init__(self, col, increasing, lo=0.0, hi=1.0):
        self.col, self.increasing, self.lo, self.hi = col, increasing, lo, hi

    def fit(self, X, y):
        self.iso = IsotonicRegression(y_min=self.lo, y_max=self.hi, increasing=self.increasing, out_of_bounds="clip").fit(X[self.col], y)
        return self

    def predict(self, X):
        return self.iso.predict(X[self.col])


class Clim:
    """Climatology from the training rows: mean of the target by station and calendar month (falls back to the
    month over all stations where a station-month has fewer than `min_n` rows). by='month' = month only (v1's baseline)."""

    def __init__(self, by="station_month", min_n=20, agg="mean"):
        self.by, self.min_n, self.agg = by, min_n, agg

    def fit(self, X, y):
        d = pd.DataFrame({"wmo": X["wmo"].values, "m": pd.DatetimeIndex(X["t"]).month, "y": np.asarray(y, float)})
        self.month = d.groupby("m").y.agg(self.agg)
        self.all = float(d.y.agg(self.agg))
        g = d.groupby(["wmo", "m"]).y.agg([self.agg, "size"])
        self.sm = g[g["size"] >= self.min_n][self.agg]
        return self

    def predict(self, X):
        m = pd.DatetimeIndex(X["t"]).month
        base = pd.Series(m).map(self.month).fillna(self.all).values
        if self.by == "month":
            return base
        idx = pd.MultiIndex.from_arrays([X["wmo"].values, m])
        sm = self.sm.reindex(idx).values
        return np.where(np.isnan(sm), base, sm)


class RawOktas:
    """Plain forecast cloud, uncalibrated: GFS total cloud % -> oktas."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.clip(X["cloud_cover"].values / 100.0 * 8.0, 0, 8)
