"""Family T: Donchian breakout (+optional EMA200 filter) with ATR initial stop and ATR trailing stop.
Protocol: select on DISC (2007-2016) by a pre-declared rule; evaluate the single selected config on CONF (2017-2022)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_m1, load_ticks, bars, summarize, PRM, PIP
from signals import atr
from core_trail import candidate_outcomes_trail
from frontier import approx_frontier


def cands(h, N, use_ema, trail_m, k_sl=2.0, ema_n=200):
    A = atr(h, 14).values
    up = h["high"].rolling(N).max().shift(1)
    lo = h["low"].rolling(N).min().shift(1)
    ema = h["close"].ewm(span=ema_n, adjust=False).mean()
    lb = h["close"] > up
    sb = h["close"] < lo
    if use_ema:
        lb = lb & (h["close"] > ema)
        sb = sb & (h["close"] < ema)
    lb_f = lb & ~lb.shift(1, fill_value=False)
    sb_f = sb & ~sb.shift(1, fill_value=False)
    trig = (lb_f | sb_f).values
    d = np.where(lb_f.values, 1.0, -1.0)
    sl = k_sl * A
    ok = trig & np.isfinite(sl) & (sl > 0)
    return h["dec_ms"].values[ok].astype(np.int64), d[ok], sl[ok], (trail_m * A)[ok]


def run(ts, bid, ask, cfg, m1):
    tf, N, use_ema, trail_m = cfg
    h = bars(m1, tf)
    dec, d, sl, tr = cands(h, N, use_ema, trail_m)
    n = len(dec)
    out = candidate_outcomes_trail(ts, bid, ask, dec, d, sl, np.zeros(n), np.zeros(n, dtype=np.int64), tr, PRM)
    df = pd.DataFrame({"dec_ms": dec, "dir": d, "sl_pips": sl / PIP, "filled": out[:, 0] > 0.5,
                       "net_pips": out[:, 5], "R": out[:, 6], "reason": out[:, 4], "exit_ts": out[:, 3]})
    return df


if __name__ == "__main__":
    stage = sys.argv[1]  # "disc" or "conf"
    grid = []
    for N in (24, 55, 120):
        for ema in (0, 1):
            for tm in (2.0, 3.0):
                grid.append(("1h", N, ema, tm))
    for N in (20, 50):
        for ema in (0, 1):
            for tm in (2.0, 3.0):
                grid.append(("4h", N, ema, tm))
    if stage == "disc":
        ts, bid, ask = load_ticks("disc"); m1 = load_m1("disc")
        rows = []
        for cfg in grid:
            df = run(ts, bid, ask, cfg, m1)
            s = summarize(df, str(cfg))
            fr = approx_frontier(df)
            rows.append({**s, **fr})
            print(cfg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**s, **fr}.items() if k != "label"}, flush=True)
        res = pd.DataFrame(rows)
        res.to_csv("/tmp/hipo_r2/results/trend_disc.csv", index=False)
        ok = res[res["n"] >= 150].sort_values("t", ascending=False)
        print("SELECTED (max disc t among n>=150):", ok.iloc[0]["label"])
    else:
        sel = sys.argv[2]
        cfg = eval(sel)
        ts, bid, ask = load_ticks("conf"); m1 = load_m1("conf")
        df = run(ts, bid, ask, cfg, m1)
        s = summarize(df, str(cfg)); fr = approx_frontier(df)
        print("CONF", cfg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**s, **fr}.items() if k != "label"})
        df.to_parquet("/tmp/hipo_r2/results/trend_conf_trades.parquet")
