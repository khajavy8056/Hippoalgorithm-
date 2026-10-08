"""Run the REAL tick engine (core.run_engine, one position at a time, equity compounding, tick-level MTM DD)
on the frozen ML diagnostic signals of CONF (2017-2022). Zero-cost and default-cost variants."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_ticks, PRM
from core import run_engine, P_SPREADMULT, P_SLIP, P_COMM, P_RISK, P_MAXSPREAD, MS_MIN, MS_DAY
PRM_G = PRM.copy(); PRM_G[P_SPREADMULT] = 0.0; PRM_G[P_SLIP] = 0.0; PRM_G[P_COMM] = 0.0; PRM_G[P_MAXSPREAD] = 0.0
ts, bid, ask = load_ticks("conf")
def monthly_stats(eq, start_ms, end_ms, bal0):
    e = pd.Series(eq[:, 1], index=pd.to_datetime(eq[:, 0].astype(np.int64), unit="ms"))
    e = e[~e.index.duplicated(keep="last")].sort_index()
    m_end = e.resample("MS").last().ffill()
    cal = pd.date_range(pd.Timestamp(start_ms, unit="ms").to_period("M").to_timestamp(), pd.Timestamp(end_ms, unit="ms").to_period("M").to_timestamp(), freq="MS")
    m_end = m_end.reindex(cal).ffill().fillna(bal0)
    prev = m_end.shift(1).fillna(bal0)
    r = m_end / prev - 1.0
    return r.mean(), r.median(), len(r)
for cfg, (s, rr, H) in [("1.5_1.0_12", (1.5, 1.0, 12)), ("2.0_1.5_48", (2.0, 1.5, 48))]:
    T = pd.read_parquet(f"/tmp/hipo_r2/results/ml_conf_{cfg}.parquet").sort_values("dec_ms")
    s_ts = T["dec_ms"].values.astype(np.int64)
    s_dir = T["dir"].values.astype(np.float64)
    s_sl = T["sl_pips"].values.astype(np.float64) * 1e-4
    s_tp = rr * s_sl
    s_hold = np.full(len(T), H * 5 * MS_MIN, dtype=np.int64)
    for label, base in [("ZERO-COST", PRM_G), ("DEFAULT COST", PRM)]:
        for risk in ([0.0025, 0.005, 0.01] if label == "ZERO-COST" else [0.005]):
            prm = base.copy(); prm[P_RISK] = risk
            tr, eq, bal, dd_abs, dd_rel, ntr = run_engine(ts, bid, ask, s_ts, s_dir, s_sl, s_tp, s_hold, prm)
            mean_m, med_m, nm = monthly_stats(eq, int(s_ts.min()), int(s_ts.max()), 10000.0)
            print(f"cfg {cfg} | {label:12s} risk {risk*100:.2f}% | trades {ntr:5d} | final equity {bal:10.0f} | "
                  f"MTM max DD {dd_rel*100:5.1f}% | mean monthly {mean_m*100:+.2f}% (median {med_m*100:+.2f}%, {nm} months)", flush=True)
