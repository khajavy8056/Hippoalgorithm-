"""ONE confirmation run on 2017-2022 with the FROZEN discovery-selected configuration.
Config (frozen): TF=5 min; TP_R=2.5; SL_BUF_ATR=0.25; MIN_STOP_ATR=2.0; CONF_BODY_ATR=0.8;
entries only 07:00-21:00 UTC (entry bar hour); risk 1.5% of equity; cost model from round 1."""
import sys, json, time
sys.path.insert(0, "/tmp/hipo_dev"); sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from common_r2 import *
from disc_grid import max_stop_atr_filter
from disc_ofat import filt
from m1bars import load_bars
import core
from core import run_engine, default_params, PTS, TR_NCOL

FROZEN = dict(TP_R=2.5, SL_BUF_ATR=0.25, CONF_BODY_ATR=0.8)
TF = 5; MIN_STOP_ATR = 2.0; RISK = 0.015; SESSION = (7, 21)

def frozen_signals(m1):
    sig = signals_for_tf(m1, TF, FROZEN)
    sig = max_stop_atr_filter(sig, MIN_STOP_ATR)
    key = m1[0]
    hh = (key[sig["k"]] // 60) % 24
    sig = filt(sig, (hh >= SESSION[0]) & (hh < SESSION[1]))
    return sig

if __name__ == "__main__":
    t0 = time.time()
    m1 = load_bars("/tmp/hipo_r2/m1_2017_2022.npz")
    key, bo, bh, bl, bc, ao, ah, al, ac, cnt = m1
    sig = frozen_signals(m1)
    print("frozen signals 2017-2022:", len(sig["k"]), " long:", int((sig["dir"]>0).sum()), " short:", int((sig["dir"]<0).sum()))
    # ---- bar-level (M1 bid/ask) ----
    tr_bar = run_bar_backtest(m1, sig, RISK)
    mb = metrics_from_trades(tr_bar)
    print("BAR-LEVEL:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mb.items()})
    pd.DataFrame(tr_bar, columns=TR_COLS).to_csv("/tmp/hipo_r2/confirm_bar_trades.csv", index=False)

    # ---- tick-level (round-1 engine on real ticks) ----
    ts = np.load("/tmp/hipo_tick_cache/full/ts.npy", mmap_mode="r")
    bid = np.load("/tmp/hipo_tick_cache/full/bid.npy", mmap_mode="r")
    ask = np.load("/tmp/hipo_tick_cache/full/ask.npy", mmap_mode="r")
    # signal -> engine inputs (distances from fill, spread-consistent with the bar executor)
    E = sig["entry"]; SL = sig["sl"]; TP = sig["tp"]; d = sig["dir"]
    hs = 0.5 * (ac[sig["k"]] - bc[sig["k"]])
    R_mid = np.abs(E - SL)
    s_sl = np.where(d < 0, (SL - E) + 2 * hs, (E - SL) + 2 * hs)
    s_tp = np.where(d < 0, (E - TP) - 2 * hs, (TP - E) - 2 * hs)
    s_ts = ((key[sig["k"]] + 1) * 60_000).astype(np.int64)        # decision time = bar close
    s_hold = np.full(len(s_ts), 1440 * 60_000, dtype=np.int64)
    order = np.argsort(s_ts, kind="stable")
    s_ts, s_dir, s_sl, s_tp, s_hold = s_ts[order], d[order].astype(np.int64), s_sl[order], s_tp[order], s_hold[order]
    s_entry_mid = E[order]
    years = list(range(2017, 2023))
    eq_bal = 10_000.0
    all_tr, all_eq = [], []
    for y in years:
        y0 = np.datetime64(f"{y}-01-01T00:00").astype("datetime64[ms]").astype(np.int64)
        y1 = np.datetime64(f"{y+1}-01-01T00:00").astype("datetime64[ms]").astype(np.int64)
        i0 = int(np.searchsorted(ts, y0)); i1 = int(np.searchsorted(ts, y1))
        ts_y = np.asarray(ts[i0:i1]); bid_y = np.asarray(bid[i0:i1]); ask_y = np.asarray(ask[i0:i1])
        q0 = int(np.searchsorted(s_ts, y0)); q1 = int(np.searchsorted(s_ts, y1))
        prm = default_params(equity=eq_bal, risk=RISK, max_lev=30.0, comm=3.5, slip_pips=0.1,
                             latency_ms=1000, fillwin_ms=30_000, max_spread_pips=2.0)
        if q1 > q0:
            tr, eqe, bal, mdd_abs, mdd_rel, ntr = run_engine(ts_y, bid_y, ask_y, s_ts[q0:q1], s_dir[q0:q1],
                                                              s_sl[q0:q1], s_tp[q0:q1], s_hold[q0:q1], prm)
            all_tr.append(tr[:ntr]); all_eq.append(eqe)
            eq_bal = bal
        print(f"  {y}: signals {q1-q0:3d}  ticks {i1-i0:,}  equity end {eq_bal:,.0f}  ({time.time()-t0:.0f}s)", flush=True)
    TRK = np.vstack(all_tr) if all_tr else np.zeros((0, TR_NCOL))
    EQK = np.vstack(all_eq) if all_eq else np.zeros((0, 2))
    np.save("/tmp/hipo_r2/confirm_tick_trades.npy", TRK)
    np.save("/tmp/hipo_r2/confirm_tick_equity.npy", EQK)
    names = {0:"entry_ts",1:"exit_ts",2:"dir",3:"entry",4:"exit",5:"lots",6:"gross",7:"comm",8:"net",9:"reason",10:"risk_usd",11:"sl",12:"tp",13:"mae",14:"mfe",15:"spread_in",16:"sig",17:"hold_min"}
    tdf = pd.DataFrame(TRK[:, :18], columns=[names[i] for i in range(18)])
    tdf.to_csv("/tmp/hipo_r2/confirm_tick_trades.csv", index=False)
    net = TRK[:, 8]; reason = TRK[:, 9].astype(int)
    eqv = EQK[:, 1]
    peak = np.maximum.accumulate(eqv)
    dd = float(np.max(1 - eqv / peak) * 100) if len(eqv) else np.nan
    print("TICK-LEVEL trades:", len(TRK), " win%:", round(float((net > 0).mean() * 100), 1) if len(net) else None,
          " final equity:", round(eq_bal, 2), " total%:", round((eq_bal / 10000 - 1) * 100, 2), " maxDD% (equity events):", round(dd, 2),
          " reasons:", {k: int((reason == k).sum()) for k in range(4)})
    print("done in %.0fs" % (time.time() - t0))
