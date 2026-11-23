"""Clear-sky probability: baselines vs a small calibrated model, scored on held-out time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .features import FEATURES

EPS = 1e-4


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [(edges[b], edges[b + 1], int((idx == b).sum()),
             float(p[idx == b].mean()) if (idx == b).any() else np.nan,
             float(y[idx == b].mean()) if (idx == b).any() else np.nan) for b in range(bins)]
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "mean_pred", "observed"])


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p),
            "auc": roc_auc_score(y, p), "ece": ece(y, p), "n": int(len(y)),
            "base_rate": float(y.mean())}


class RawCloudBaseline:
    """What you get by reading the forecast's total cloud cover, calibrated (isotonic)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.iso = IsotonicRegression(y_min=0, y_max=1, increasing=False, out_of_bounds="clip")
        self.iso.fit(X["cloud_cover"].values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(X["cloud_cover"].values)


class NaiveThreshold:
    """What most weather apps imply: 'clear' if forecast cloud cover is under 25%, else 'not clear'."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(X["cloud_cover"].values < 25, 0.9, 0.1)  # a hard call, softened to avoid log(0)


class Climatology:
    """Clear-night rate by month only."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        m = pd.Series(y, index=X.index).groupby(X.index.month).mean()
        self.rate, self.mean = m, float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.index.month.map(self.rate).fillna(self.mean).to_numpy(dtype=float)


class Logistic:
    def fit(self, X, y):
        self.mu, self.sd = X[FEATURES].mean(), X[FEATURES].std().replace(0, 1)
        self.m = LogisticRegression(max_iter=2000, C=1.0)
        self.m.fit((X[FEATURES] - self.mu) / self.sd, y)
        return self

    def predict(self, X):
        return self.m.predict_proba((X[FEATURES] - self.mu) / self.sd)[:, 1]


class Boosted:
    """Gradient-boosted trees, then isotonic calibration on a later slice of the training set."""

    def fit(self, X, y, calib_frac: float = 0.25):
        n = len(X)
        cut = int(n * (1 - calib_frac))  # time-ordered: calibrate on the latest part
        self.m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                min_samples_leaf=40, l2_regularization=1.0,
                                                random_state=0)
        self.m.fit(X[FEATURES].iloc[:cut], y[:cut])
        raw = self.m.predict_proba(X[FEATURES].iloc[cut:])[:, 1]
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, y[cut:])
        return self

    def predict(self, X):
        return self.iso.predict(self.m.predict_proba(X[FEATURES])[:, 1])


MODELS = {"climatology": Climatology, "naive threshold": NaiveThreshold,
          "raw forecast, calibrated": RawCloudBaseline,
          "logistic": Logistic, "boosted + calibrated": Boosted}
