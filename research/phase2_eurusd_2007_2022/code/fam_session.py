"""Family S: time-of-day effects on H1 decision bars (UTC).
S1 unconditional: enter at hour H (bar close), direction +1/-1, time-stop k hours.
S2 conditional: sign of return over the last k1 hours predicts (momentum or reversal) over next k2 hours.
Stop = 3*ATR(H1) (catastrophe only). Selection on DISC by t-stat of net pips; then one CONF evaluation."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_m1, load_ticks, bars, PRM, PIP
from core import candidate_outcomes, MS_HOUR
from signals import atr


def prep(name):
    m1 = load_m1(name)
    h = bars(m1, "1h")
    dec = h["dec_ms"].values.astype(np.int64)
    close = pd.Series(h["close"].values, index=dec)
    A = pd.Series(atr(h, 14).values, index=dec)
    hod = (dec // MS_HOUR) % 24
    return dec, close, A, hod


def run_cfg(ts, bid, ask, prep_obj, cfg):
    dec, close, A, hod = prep_obj
    kind, H, k2, d_or_mode, k1 = cfg
    sel = hod == H
    base = dec[sel]
    if kind == "uncond":
        d = np.full(len(base), float(d_or_mode))
    else:
        prev = close.reindex(base - k1 * MS_HOUR).values
        now = close.reindex(base).values
        r = np.sign(now - prev)
        d = r if d_or_mode == "mom" else -r
    a = A.reindex(base).values
    ok = np.isfinite(a) & (a > 0) & (d != 0) & np.isfinite(d)
    dec_c = base[ok]
    d_c = d[ok].astype(np.float64)
    sl = 3.0 * a[ok]
    hold = np.full(len(dec_c), k2 * MS_HOUR, dtype=np.int64)
    out = candidate_outcomes(ts, bid, ask, dec_c, d_c, sl, np.zeros(len(dec_c)), hold, PRM)
    f = out[:, 0] > 0.5
    p = out[f, 5]
    n = len(p)
    if n < 30:
        return {"n": n, "mean_pips": np.nan, "t_pips": np.nan}
    return {"n": n, "mean_pips": p.mean(), "t_pips": p.mean() / (p.std(ddof=1) / np.sqrt(n)),
            "hit": (p > 0).mean(), "dec_first": int(dec_c[f].min()), "dec_last": int(dec_c[f].max())}


def grid():
    G = []
    for H in range(24):
        for k2 in (1, 2, 4, 8):
            for d in (1, -1):
                G.append(("uncond", H, k2, d, 0))
    for H in range(24):
        for k1 in (1, 4, 12, 24):
            for k2 in (1, 4, 8):
                for mode in ("mom", "rev"):
                    G.append(("cond", H, k2, mode, k1))
    return G


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "disc":
        ts, bid, ask = load_ticks("disc")
        P = prep("disc")
        rows = []
        for cfg in grid():
            r = run_cfg(ts, bid, ask, P, cfg)
            rows.append({"cfg": str(cfg), **r})
        res = pd.DataFrame(rows)
        res.to_csv("/tmp/hipo_r2/results/session_disc.csv", index=False)
        ok = res[res["n"] >= 500].sort_values("t_pips", ascending=False)
        print(ok.head(15).to_string())
        print("SELECTED:", ok.iloc[0]["cfg"], "t=", round(ok.iloc[0]["t_pips"], 2), "n=", ok.iloc[0]["n"])
        print("number of configs with t>2 on DISC:", int((ok["t_pips"] > 2).sum()), "of", len(ok))
    else:
        cfg = eval(sys.argv[2])
        ts, bid, ask = load_ticks("conf")
        P = prep("conf")
        r = run_cfg(ts, bid, ask, P, cfg)
        print("CONF", cfg, r)
