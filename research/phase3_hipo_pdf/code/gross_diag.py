import sys
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np
from numba import njit
from common_r2 import *
from disc_grid import load_years, max_stop_atr_filter

@njit(cache=True)
def gross_outcomes(mh, ml, k_arr, d_arr, sl_arr, tp_arr, maxbars):
    n = mh.shape[0]
    out = np.full((k_arr.shape[0], 3), np.nan)   # R, reason(0 SL,1 TP,2 timeout), bars held
    for q in range(k_arr.shape[0]):
        k = k_arr[q]; d = d_arr[q]; sl = sl_arr[q]; tp = tp_arr[q]
        R = abs(tp_arr[q] - sl_arr[q]) / 2.5
        out[q, 0] = np.nan
        j = k + 1; done = False
        while j < n and j - k <= maxbars:
            if d < 0:
                if mh[j] >= sl:
                    out[q, 0] = -1.0; out[q, 1] = 0; out[q, 2] = j - k; done = True; break
                if ml[j] <= tp:
                    out[q, 0] = 2.5; out[q, 1] = 1; out[q, 2] = j - k; done = True; break
            else:
                if ml[j] <= sl:
                    out[q, 0] = -1.0; out[q, 1] = 0; out[q, 2] = j - k; done = True; break
                if mh[j] >= tp:
                    out[q, 0] = 2.5; out[q, 1] = 1; out[q, 2] = j - k; done = True; break
            j += 1
        if not done:
            out[q, 1] = 2
    return out

if __name__ == "__main__":
    m1 = load_years(list(range(2007, 2017)))
    key, bo, bh, bl, bc, ao, ah, al, ac, cnt = m1
    for tf in (1, 5, 15):
        for msa in (0.0, 2.0):
            sig = max_stop_atr_filter(signals_for_tf(m1, tf, dict(TP_R=2.5, SL_BUF_ATR=0.25)), msa)
            # mid M1 arrays for execution-free gross test; entry on the M1 bar that closes the TF bar
            mh = (bh + ah) / 2; ml = (bl + al) / 2
            o = gross_outcomes(mh, ml, sig["k"], sig["dir"], sig["sl"], sig["tp"], 1440)
            R = o[:, 0]; valid = ~np.isnan(R)
            wins = (o[valid, 1] == 1).mean() * 100
            print(f"TF={tf:2d} MSA={msa}: setups={len(R)} resolved={valid.sum()} gross_mean_R={np.nanmean(R):+.3f} "
                  f"TP%={wins:.1f} (breakeven at 2.5R = 28.6%)  timeouts={int((o[:,1]==2).sum())}  "
                  f"median_stop_pips={np.median(np.abs(sig['entry']-sig['sl']))*1e4:.2f}")
