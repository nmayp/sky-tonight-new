"""Metrics, night-level bootstrap and reliability points for the v2 evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

EPS = 1e-4


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def binary_scores(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    auc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    return {"brier": float(np.mean((p - y) ** 2)), "logloss": float(log_loss(y, p)), "auc": auc, "ece": ece(y, p),
            "n": int(len(y)), "base_rate": float(y.mean())}


def oktas_scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "bias": float(np.mean(p - y)), "n": int(len(y)), "mean_obs": float(y.mean())}


def _night_sums(nights: np.ndarray, per_row: dict[str, np.ndarray]):
    codes, uniq = pd.factorize(nights)
    n = np.bincount(codes, minlength=len(uniq)).astype(float)
    S = {k: np.bincount(codes, weights=v, minlength=len(uniq)) for k, v in per_row.items()}
    return n, S


def bootstrap(nights, per_row: dict[str, np.ndarray], pairs: list[tuple[str, str]] = (), reps: int = 2000, seed: int = 0) -> dict:
    """95% intervals for the mean of each per-row loss and for paired differences, resampling whole nights.

    per_row: name -> per-row loss (squared error for Brier, absolute error for MAE, log loss ...)."""
    n, S = _night_sums(np.asarray(nights), per_row)
    rng = np.random.default_rng(seed)
    K = len(n)
    cnt = rng.multinomial(K, np.full(K, 1.0 / K), size=reps).astype(float)       # reps x K night counts
    den = cnt @ n
    out = {"point": {k: float(S[k].sum() / n.sum()) for k in S}, "ci": {}, "diff": {}}
    means = {k: (cnt @ S[k]) / den for k in S}
    for k, m in means.items():
        out["ci"][k] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    for a, b in pairs:
        d = means[a] - means[b]
        out["diff"][f"{a}-{b}"] = {"point": float(S[a].sum() / n.sum() - S[b].sum() / n.sum()),
                                   "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                   "share_below_zero": float(np.mean(d < 0))}
    out["n_nights"] = int(K)
    return out


def reliability_points(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"p": round(float(p[idx == b].mean()), 4), "obs": round(float(y[idx == b].mean()), 4), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]
