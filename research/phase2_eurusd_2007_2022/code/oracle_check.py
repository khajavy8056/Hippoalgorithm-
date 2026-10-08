"""Engine sanity checks on DISC (2011 sample):
(1) ORACLE: direction = sign of the true mid move over the next H bars -> must be strongly positive.
(2) ANTI-ORACLE: opposite sign -> must be strongly negative.
(3) RANDOM direction -> should be negative by roughly the cost hurdle.
(4) Same signals with gross (no-cost) fills -> mean gross R should be ~ +/- the oracle edge."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_m1, load_ticks, bars, PRM
from core import candidate_outcomes, MS_MIN, P_SPREADMULT, P_SLIP, P_COMM, P_MAXSPREAD
from signals import atr
PRM_G = PRM.copy(); PRM_G[P_SPREADMULT] = 0.0; PRM_G[P_SLIP] = 0.0; PRM_G[P_COMM] = 0.0; PRM_G[P_MAXSPREAD] = 0.0
ts, bid, ask = load_ticks("disc")
m1 = load_m1("disc")
b5 = bars(m1, "5min")
A = atr(b5, 14).values
c = b5["close"].values
H = 12
fut = np.full(len(c), np.nan); fut[:-H] = c[H:] - c[:-H]
dec = b5["dec_ms"].values.astype(np.int64)
yr = pd.to_datetime(dec // 1000, unit="s").year
sel = (yr == 2011) & np.isfinite(fut) & np.isfinite(A) & (A > 0)
dec_s, A_s, fut_s = dec[sel], A[sel], fut[sel]
sl = 1.5 * A_s; tp = 1.0 * sl; hold = np.full(len(dec_s), H * 5 * MS_MIN, dtype=np.int64)
def run(d, prm):
    o = candidate_outcomes(ts, bid, ask, dec_s, d.astype(float), sl, tp, hold, prm)
    f = o[:, 0] > 0.5
    return f.sum(), np.nanmean(o[f, 6]), np.nanmean(o[f, 5])
orc = np.sign(fut_s)
rng = np.random.default_rng(1)
rnd = rng.choice([-1.0, 1.0], len(dec_s))
for name, d in [("ORACLE (perfect direction)", orc), ("ANTI-ORACLE", -orc), ("RANDOM direction", rnd)]:
    n, R_net, p_net = run(d, PRM)
    n2, R_g, p_g = run(d, PRM_G)
    print(f"{name:28s} n={n:7d}  NET meanR={R_net:+.3f} (pips {p_net:+.2f})   GROSS meanR={R_g:+.3f} (pips {p_g:+.2f})")
