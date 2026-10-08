"""Discovery-only grid (2007-2016). Pre-declared before running:
   TF in {1,5,15} min; TP_R in {2.5,3.0}; SL_BUF_ATR in {0.10,0.25}; MIN_STOP_ATR in {0,2.0} (skip setups whose
   stop distance < m * ATR_TF); risk in {0.25,0.5,1,2}% per trade. Selection rule (discovery only):
   maximise monthly geometric return subject to max realised DD <= 10%; if none qualifies, maximise return/DD."""
import sys, json, itertools, time
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from common_r2 import *
from m1bars import load_bars

def year_range_minutes(y):
    t0 = np.datetime64(f"{y}-01-01T00:00").astype("datetime64[m]").astype(np.int64)
    t1 = np.datetime64(f"{y+1}-01-01T00:00").astype("datetime64[m]").astype(np.int64)
    return t0, t1

def load_years(years):
    parts = []
    for y in years:
        cols = load_bars(f"/tmp/hipo_r2/bars/m1_{y}.npz")
        t0, t1 = year_range_minutes(y)
        k = cols[0]
        sel = (k >= t0) & (k < t1)
        parts.append([c[sel] for c in cols])
        print(f"  {y}: kept {sel.sum():,} of {len(k):,} bars")
    cols = [np.concatenate([p[j] for p in parts]) for j in range(10)]
    assert np.all(np.diff(cols[0]) > 0), "keys not strictly increasing"
    return cols

def max_stop_atr_filter(sig, min_stop_atr):
    if sig is None or min_stop_atr <= 0:
        return sig
    R = np.abs(sig["entry"] - sig["sl"])
    keep = R >= min_stop_atr * sig["atr"]
    out = {}
    for kk, vv in sig.items():
        if kk in ("tf", "n_tf_bars"):
            out[kk] = vv
        else:
            out[kk] = vv[keep]
    return out

if __name__ == "__main__":
    t0 = time.time()
    m1 = load_years(list(range(2007, 2017)))
    print("discovery M1 bars:", len(m1[0]), "time %.1fs" % (time.time() - t0))
    RISKS = [0.0025, 0.005, 0.01, 0.02]
    rows = []
    grid = list(itertools.product([1, 5, 15], [2.5, 3.0], [0.10, 0.25], [0.0, 2.0]))
    for tf, tpr, sbuf, msa in grid:
        kw = dict(TP_R=tpr, SL_BUF_ATR=sbuf)
        sig = signals_for_tf(m1, tf, kw)
        sig = max_stop_atr_filter(sig, msa)
        n_sig = 0 if sig is None else len(sig["k"])
        for risk in RISKS:
            tr = run_bar_backtest(m1, sig, risk)
            met = metrics_from_trades(tr)
            row = dict(TF=tf, TP_R=tpr, SL_BUF_ATR=sbuf, MIN_STOP_ATR=msa, risk_pct=risk * 100, signals=n_sig)
            row.update(met)
            rows.append(row)
        best = [r for r in rows if r["TF"] == tf and r["TP_R"] == tpr and r["SL_BUF_ATR"] == sbuf and r["MIN_STOP_ATR"] == msa]
        print(f"TF={tf:2d} TP={tpr} SLB={sbuf} MSA={msa}: signals={n_sig:6d} | 0.5%: trades={best[1].get('trades',0)} "
              f"meanR={best[1].get('mean_R', float('nan')):.3f} win={best[1].get('win_pct', float('nan')):.1f}% "
              f"mo={best[1].get('monthly_geo_pct', float('nan')):.3f}% DD={best[1].get('max_dd_pct', float('nan')):.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/tmp/hipo_r2/disc_grid_results.csv", index=False)
    print("saved", len(df), "rows")
