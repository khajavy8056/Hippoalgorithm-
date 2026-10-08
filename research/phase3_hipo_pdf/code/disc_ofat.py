"""One-factor-at-a-time exploration on DISCOVERY (2007-2016) only, around the base TF=5 / TP 2.5R / SL buf 0.25 / min stop 2 ATR.
Pre-declared: 9 variants + base = 10 trials. Filters that are optional in the PDF (time divergence) or
implementation choices (session window, thresholds) are tested one at a time."""
import sys, time
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from common_r2 import *
from disc_grid import load_years, max_stop_atr_filter
from m1bars import weekday_tow

def filt(sig, mask):
    out = {}
    for kk, vv in sig.items():
        out[kk] = vv if kk in ("tf", "n_tf_bars") else vv[mask]
    return out

def session_mask(m1, sig, h0=7, h1=21):
    key = m1[0]
    hh = (key[sig["k"]] // 60) % 24
    return (hh >= h0) & (hh < h1)

if __name__ == "__main__":
    m1 = load_years(list(range(2007, 2017)))
    base = dict(TP_R=2.5, SL_BUF_ATR=0.25)
    variants = [
        ("base (TF5, TP2.5, SLB0.25, MSA2)", {}, None),
        ("V1 require time divergence (BC bars > AB bars)", {}, "tdiv"),
        ("V2 confirmation body >= 0.8 ATR", dict(CONF_BODY_ATR=0.8), None),
        ("V3 signal range >= 1.2 ATR", dict(SIG_MIN_RANGE=1.2), None),
        ("V4 entries only 07:00-21:00 UTC", {}, "session"),
        ("V5 confirmation wait <= 5 bars", dict(CONF_MAX_WAIT=5), None),
        ("V6 confirmation wait <= 30 bars", dict(CONF_MAX_WAIT=30), None),
        ("V7 B position within signal 0.4-0.6", dict(SIG_POS_LO=0.4, SIG_POS_HI=0.6), None),
        ("V8 signal upper wick <= 0.20 of range", dict(SIG_MAX_WICK=0.20), None),
        ("V9 AB noise candles = 0", dict(AB_MAX_NOISE=0), None),
    ]
    rows = []
    for name, extra, flt in variants:
        kw = dict(base); kw.update(extra)
        sig = signals_for_tf(m1, 5, kw)
        sig = max_stop_atr_filter(sig, 2.0)
        if sig is not None and flt == "tdiv":
            raw = sig["raw"]
            sig = filt(sig, raw[:, 13] == 1)
        if sig is not None and flt == "session":
            sig = filt(sig, session_mask(m1, sig))
        n = 0 if sig is None else len(sig["k"])
        # gross (mid, no costs)
        if n:
            mh = (m1[2] + m1[6]) / 2; ml = (m1[3] + m1[7]) / 2
            from gross_diag import gross_outcomes
            o = gross_outcomes(mh, ml, sig["k"], sig["dir"], sig["sl"], sig["tp"], 1440)
            gR = float(np.nanmean(o[:, 0])) if np.any(~np.isnan(o[:, 0])) else np.nan
        else:
            gR = np.nan
        for risk in (0.0025, 0.005, 0.01, 0.02):
            tr = run_bar_backtest(m1, sig, risk)
            met = metrics_from_trades(tr)
            row = dict(variant=name, risk_pct=risk*100, signals=n, gross_mean_R=gR); row.update(met)
            rows.append(row)
        r5 = [r for r in rows if r["variant"] == name and abs(r["risk_pct"]-0.5) < 1e-9][0]
        print(f"{name:52s} n={n:4d} gross_R={gR:+.3f} | 0.5%: trades={r5.get('trades',0):4d} meanR={r5.get('mean_R',np.nan):+.3f} "
              f"mo={r5.get('monthly_geo_pct',np.nan):+.3f}% DD={r5.get('max_dd_pct',np.nan):.2f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/tmp/hipo_r2/disc_ofat_results.csv", index=False)
    print("saved", len(df))
