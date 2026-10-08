import sys, glob, os
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np
import pandas as pd
from numba import njit
from m1bars import load_bars
from hipo_core import scan_direction, params_vector, atr_wilder, OUT_COLS
from exec_bar import exec_bars, exec_params, mid_bars, TR_COLS

@njit(cache=True)
def resample_bidask(key, bo, bh, bl, bc, ao, ah, al, ac, tf):
    n = key.shape[0]
    rk = np.empty(n, np.int64)
    R = np.empty((n, 8))
    m = -1; cur = -1
    for i in range(n):
        bk = key[i] // tf
        if bk != cur:
            m += 1; cur = bk
            rk[m] = bk * tf                     # bucket start minute
            R[m, 0] = bo[i]; R[m, 1] = bh[i]; R[m, 2] = bl[i]; R[m, 3] = bc[i]
            R[m, 4] = ao[i]; R[m, 5] = ah[i]; R[m, 6] = al[i]; R[m, 7] = ac[i]
        else:
            if bh[i] > R[m, 1]: R[m, 1] = bh[i]
            if bl[i] < R[m, 2]: R[m, 2] = bl[i]
            R[m, 3] = bc[i]
            if ah[i] > R[m, 5]: R[m, 5] = ah[i]
            if al[i] < R[m, 6]: R[m, 6] = al[i]
            R[m, 7] = ac[i]
    m += 1
    return rk[:m], R[:m, 0].copy(), R[:m, 1].copy(), R[:m, 2].copy(), R[:m, 3].copy(), \
        R[:m, 4].copy(), R[:m, 5].copy(), R[:m, 6].copy(), R[:m, 7].copy()

def load_discovery():
    parts = []
    for y in range(2007, 2017):
        p = f"/tmp/hipo_r2/bars/m1_{y}.npz"
        parts.append(load_bars(p))
    cols = [np.concatenate([p[j] for p in parts]) for j in range(10)]
    return cols

def load_confirm():
    return load_bars("/tmp/hipo_r2/m1_2017_2022.npz")

def signals_for_tf(m1, tf, prm_kw):
    """Detect HIPO setups on TF-minute mid bars; return signals mapped to M1 entry indices."""
    key, bo, bh, bl, bc, ao, ah, al, ac, cnt = m1
    rk, RBO, RBH, RBL, RBC, RAO, RAH, RAL, RAC = resample_bidask(key, bo, bh, bl, bc, ao, ah, al, ac, tf)
    mo, mh, ml, mc = (RBO + RAO) / 2, (RBH + RAH) / 2, (RBL + RAL) / 2, (RBC + RAC) / 2
    atr = atr_wilder(mh, ml, mc, 14)
    P = params_vector(**prm_kw)
    rs = []
    for d in (+1, -1):
        r = scan_direction(mo, mh, ml, mc, atr, P, d)
        if len(r):
            r = np.column_stack([r, np.full(len(r), -d, float)])   # col 14: trade dir (-1 short, +1 long)
            rs.append(r)
    if not rs:
        return None
    S = np.vstack(rs)
    S = S[np.argsort(S[:, 0], kind="stable")]
    # map TF-bar entry index -> M1 index of the last minute of that bucket (bar close time)
    close_min = rk[S[:, 0].astype(np.int64)] + tf - 1
    k_m1 = np.searchsorted(key, close_min, side="right") - 1
    keep = (k_m1 >= 0) & (key[np.clip(k_m1, 0, len(key)-1)] == close_min)
    S = S[keep]; k_m1 = k_m1[keep]
    order = np.argsort(k_m1, kind="stable")
    return dict(k=k_m1[order].astype(np.int64), dir=S[order, 14].astype(np.int64),
                entry=S[order, 5], sl=S[order, 6], tp=S[order, 7], ab=S[order, 8], atr=S[order, 9],
                tf=tf, n_tf_bars=len(rk), raw=S[order])

def metrics_from_trades(tr, eq0=10_000.0):
    if len(tr) == 0:
        return dict(trades=0)
    pnl = tr[:, 8]; R = tr[:, 9]; reason = tr[:, 10].astype(int)
    eq = np.concatenate([[eq0], tr[:, 11]])
    peak = np.maximum.accumulate(eq)
    dd = np.max(1 - eq / peak) * 100
    wins = pnl[pnl > 0].sum(); losses = -pnl[pnl < 0].sum()
    # monthly geometric returns from equity at last trade of each month
    exit_ts = pd.to_datetime(tr[:, 1].astype(np.int64), unit="m")
    s = pd.Series(tr[:, 11], index=exit_ts)
    me = s.groupby(s.index.to_period("M")).last()
    prev = np.concatenate([[eq0], me.values[:-1]])
    mret = me.values / prev - 1
    n_months = len(me)
    total = (tr[-1, 11] / eq0 - 1) * 100
    monthly_geo = ((tr[-1, 11] / eq0) ** (1 / max(n_months, 1)) - 1) * 100 if tr[-1, 11] > 0 else -100
    return dict(trades=len(tr), win_pct=float((pnl > 0).mean() * 100), mean_R=float(R.mean()),
                pf=float(wins / losses) if losses > 0 else np.nan, total_pct=float(total),
                monthly_geo_pct=float(monthly_geo), max_dd_pct=float(dd),
                worst_month_pct=float(mret.min() * 100), pos_months_pct=float((mret > 0).mean() * 100),
                n_months=int(n_months), tp_share=float((reason == 1).mean() * 100),
                sl_share=float((reason == 0).mean() * 100))

def run_bar_backtest(m1, sig, risk, max_spread=2.0, max_hold=1440):
    prm = exec_params(risk=risk, max_spread_pips=max_spread, max_hold_min=max_hold)
    key, bo, bh, bl, bc, ao, ah, al, ac, cnt = m1
    if sig is None:
        return np.zeros((0, 12))
    tr = exec_bars(key, bo, bh, bl, bc, ao, ah, al, ac, sig["k"], sig["dir"], sig["entry"], sig["sl"], sig["tp"], prm)
    return tr
