"""Approximate portfolio frontier from a per-trade R sequence (screening only; the tick engine is used for finalists).
Each trade risks r of equity; equity compounds at trade exit; monthly mean = mean over calendar months (exit month) of sum r*R."""
import numpy as np, pandas as pd

RISKS = [0.0025, 0.005, 0.01, 0.02, 0.03]


def _path(df, r):
    f = df[df["filled"]].sort_values("exit_ts")
    if len(f) == 0:
        return np.array([0.0]), pd.Series(dtype=float)
    R = f["R"].values
    eq = np.cumprod(1.0 + r * R)
    eq = np.concatenate([[1.0], eq])
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / peak
    m = pd.to_datetime(f["exit_ts"].values // 1000, unit="s").to_period("M")
    mon = pd.Series(r * R).groupby(np.asarray(m)).sum()
    return dd, mon


def approx_frontier(df):
    out = {}
    if len(df[df["filled"]]) < 5:
        return {"DD10_monthly": np.nan, "risk_at_DD10": np.nan, "monthly_at_0.5pct": np.nan, "DD_at_0.5pct": np.nan}
    # calendar span of the stage (months incl. zero-trade months)
    t = df[df["filled"]]["dec_ms"].values
    span_m = (pd.Timestamp(int(t.max()) // 1000, unit="s").to_period("M") - pd.Timestamp(int(t.min()) // 1000, unit="s").to_period("M")).n + 1
    best = (0.0, np.nan)
    for r in RISKS:
        dd, mon = _path(df, r)
        mmean = mon.sum() / span_m  # mean monthly simple return incl. empty months
        mdd = float(np.max(dd))
        if mdd <= 0.10 and mmean > best[0]:
            best = (mmean, r)
        if abs(r - 0.005) < 1e-12:
            out["monthly_at_0.5pct"] = mmean
            out["DD_at_0.5pct"] = mdd
    out["DD10_monthly"] = best[0]
    out["risk_at_DD10"] = best[1]
    return out
