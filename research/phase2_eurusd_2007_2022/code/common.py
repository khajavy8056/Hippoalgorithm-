"""Shared research helpers (outside the repo): stores, bars, stateless tick evaluator, metrics."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev")
from core import candidate_outcomes, default_params, PIP, PTS, MS_MIN, MS_HOUR, MS_DAY
from signals import resample_bars, atr, rsi, adx

STORES = {
    "disc": ("/tmp/hipo_data_0716/full", "/tmp/hipo_r2/m1_disc.parquet"),   # 2007-2016 (discovery)
    "conf": ("/tmp/hipo_tick_cache/full", "/tmp/hipo_r2/m1_conf.parquet"),  # 2017-2022-07 (confirmation)
}
PRM = default_params()  # equity 10k, comm 3.5/lot/side, slip 0.1 pip, latency 1 s, spread filter 2 pips, Fri cut-off


def load_ticks(name):
    d = STORES[name][0]
    return (np.load(f"{d}/ts.npy", mmap_mode="r"), np.load(f"{d}/bid.npy", mmap_mode="r"),
            np.load(f"{d}/ask.npy", mmap_mode="r"))


def load_m1(name):
    return pd.read_parquet(STORES[name][1])


def bars(m1, rule):
    return resample_bars(m1, rule)


def evaluate(name_or_arrays, c_ts, c_dir, c_sl, c_tp=None, c_hold=None, prm=PRM):
    """Stateless tick-level outcome of each candidate (one per decision; no portfolio overlap).
    c_sl in price units (>0), c_tp price units (0 = none), c_hold ms (0 = none).
    Returns DataFrame with filled, net_pips, R, reason, exit_ts, spread_pips."""
    if isinstance(name_or_arrays, str):
        ts, bid, ask = load_ticks(name_or_arrays)
    else:
        ts, bid, ask = name_or_arrays
    c_ts = np.asarray(c_ts, dtype=np.int64)
    c_dir = np.asarray(c_dir, dtype=np.float64)
    c_sl = np.asarray(c_sl, dtype=np.float64)
    c_tp = np.zeros(len(c_ts)) if c_tp is None else np.asarray(c_tp, dtype=np.float64)
    c_hold = np.zeros(len(c_ts), dtype=np.int64) if c_hold is None else np.asarray(c_hold, dtype=np.int64)
    out = candidate_outcomes(ts, bid, ask, c_ts, c_dir, c_sl, c_tp, c_hold, prm)
    df = pd.DataFrame({"dec_ms": c_ts, "dir": c_dir, "sl_pips": c_sl / PIP, "filled": out[:, 0] > 0.5,
                       "net_pips": out[:, 5], "R": out[:, 6], "reason": out[:, 4],
                       "exit_ts": out[:, 3], "spread_pips": out[:, 7]})
    return df


def summarize(df, label=""):
    f = df[df["filled"]]
    n = len(f)
    if n < 2:
        return {"label": label, "n": n, "mean_R": np.nan, "t": np.nan, "mean_pips": np.nan, "hit": np.nan, "PF": np.nan}
    r = f["R"].values
    p = f["net_pips"].values
    gp = p[p > 0].sum(); gl = -p[p < 0].sum()
    yrs = pd.to_datetime(f["dec_ms"].values // 1000, unit="s").year
    by = pd.Series(r).groupby(np.asarray(yrs)).agg(["mean", "size"])
    by = by[by["size"] >= 20]
    return {"label": label, "n": n, "mean_R": r.mean(), "sd_R": r.std(ddof=1),
            "t": r.mean() / (r.std(ddof=1) / np.sqrt(n)), "mean_pips": p.mean(),
            "hit": (p > 0).mean(), "PF": gp / gl if gl > 0 else np.inf,
            "yrs_pos": f"{int((by['mean'] > 0).sum())}/{len(by)}" if len(by) else "-"}
