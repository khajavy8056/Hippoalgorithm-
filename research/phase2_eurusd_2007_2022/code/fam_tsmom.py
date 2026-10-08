"""Family D: daily time-series momentum (sign of L-day return), decided at the daily close.
Stop 3*ATR20(daily), no TP, time-stop H days. Weekend force-close DISABLED for this family (swing holds);
entries still use the engine's spread filter (2 pips). One position at a time (greedy)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_m1, load_ticks, bars, PRM, summarize
from core import candidate_outcomes, MS_DAY, P_FCLOSE_TOW
from signals import atr
from frontier import approx_frontier
PRM_SW = PRM.copy(); PRM_SW[P_FCLOSE_TOW] = 1e18   # no weekend force-close


def cands(name, L, H):
    m1 = load_m1(name)
    d1 = bars(m1, "1D")
    A = atr(d1, 20).values
    c = d1["close"].values
    mom = np.full(len(c), np.nan)
    mom[L:] = np.sign(c[L:] - c[:-L])
    ok = np.isfinite(mom) & (mom != 0) & np.isfinite(A) & (A > 0)
    return d1["dec_ms"].values[ok].astype(np.int64), mom[ok], 3.0 * A[ok], np.full(ok.sum(), H * MS_DAY, dtype=np.int64)


def run(name, L, H):
    dec, d, sl, hold = cands(name, L, H)
    ts, bid, ask = load_ticks(name)
    o = candidate_outcomes(ts, bid, ask, dec, d.astype(float), sl, np.zeros(len(dec)), hold, PRM_SW)
    df = pd.DataFrame({"dec_ms": dec, "dir": d, "filled": o[:, 0] > 0.5, "net_pips": o[:, 5], "R": o[:, 6], "exit_ts": o[:, 3]})
    f = df[df["filled"]].sort_values("dec_ms", kind="stable")
    # one position at a time
    keep = []; last = -1
    for i, (t, e) in enumerate(zip(f["dec_ms"].values, f["exit_ts"].values)):
        if t >= last:
            keep.append(i); last = e
    return f.iloc[keep]


if __name__ == "__main__":
    stage = sys.argv[1]
    grid = [(L, H) for L in (20, 60, 120, 250) for H in (20, 60)]
    if stage == "disc":
        rows = []
        for (L, H) in grid:
            T = run("disc", L, H)
            s = summarize(T.assign(filled=True), f"L{L}_H{H}")
            fr = approx_frontier(T.assign(filled=True))
            rows.append({**s, **fr, "L": L, "H": H})
            print(L, H, {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in {**s}.items() if k != "label"}, flush=True)
        res = pd.DataFrame(rows)
        res.to_csv("/tmp/hipo_r2/results/tsmom_disc.csv", index=False)
        ok = res[res["n"] >= 30].sort_values("t", ascending=False)
        print("SELECTED:", ok.iloc[0]["label"], "t=", round(ok.iloc[0]["t"], 2))
    else:
        L, H = eval(sys.argv[2])
        T = run("conf", L, H)
        s = summarize(T.assign(filled=True), f"CONF L{L}_H{H}")
        print("CONF", {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()},
              approx_frontier(T.assign(filled=True)))
