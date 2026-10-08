"""Stateless tick-level evaluator with an optional TRAILING stop (distance c_trail, price units).
Same conventions as core.candidate_outcomes: long fills at ask (+slip), exits at bid (-slip)."""
import sys
import numpy as np
from numba import njit
sys.path.insert(0, "/tmp/hipo_dev")
from core import (PTS, PIP_VALUE_LOT, P_LAT, P_FILLWIN, P_SLIP, P_COMM, P_FCLOSE_TOW, P_SPREADMULT,
                  _bsearch_left, _tow, _eff, _entry_ok)


@njit(cache=True)
def candidate_outcomes_trail(ts, bid, ask, c_ts, c_dir, c_sl, c_tp, c_hold, c_trail, prm):
    n = ts.shape[0]
    nc = c_ts.shape[0]
    lat = prm[P_LAT]
    fw = prm[P_FILLWIN]
    slip = prm[P_SLIP]
    comm_pips = 2.0 * prm[P_COMM] / PIP_VALUE_LOT
    fclose = prm[P_FCLOSE_TOW]
    out = np.full((nc, 8), np.nan)
    for c in range(nc):
        t_ready = c_ts[c] + lat
        k = _bsearch_left(ts, t_ready, 0, n)
        if k >= n:
            out[c, 0] = 0.0
            continue
        m = k
        filled = False
        fill = 0.0
        while m < n and ts[m] - t_ready <= fw:
            a, b = _eff(ask[m] / PTS, bid[m] / PTS, prm[P_SPREADMULT])
            if _entry_ok(a, b, ts[m], prm):
                d = c_dir[c]
                fill = a + slip if d > 0 else b - slip
                filled = True
                break
            m += 1
        if not filled or c_sl[c] <= 0:
            out[c, 0] = 0.0
            continue
        d = c_dir[c]
        sl_d = c_sl[c]
        tp_d = c_tp[c]
        hold = c_hold[c]
        tr_d = c_trail[c]
        sl_lvl = fill - sl_d if d > 0 else fill + sl_d
        tp_lvl = (fill + tp_d if d > 0 else fill - tp_d) if tp_d > 0 else 0.0
        hold_end = (ts[m] + hold) if hold > 0 else (1 << 62)
        best = fill
        q = m + 1
        exit_px = np.nan
        reason = -1
        exit_ts = 0
        while q < n:
            t = ts[q]
            ae, be = _eff(ask[q] / PTS, bid[q] / PTS, prm[P_SPREADMULT])
            if d > 0:
                cur = be
                if cur <= sl_lvl:
                    exit_px = (cur if cur < sl_lvl else sl_lvl) - slip
                    reason = 0
                elif tp_lvl > 0.0 and cur >= tp_lvl:
                    exit_px = tp_lvl
                    reason = 1
                elif t >= hold_end:
                    exit_px = cur - slip
                    reason = 2
                elif _tow(t) >= fclose:
                    exit_px = cur - slip
                    reason = 3
                elif cur > best:
                    best = cur
                    if tr_d > 0 and best - tr_d > sl_lvl:
                        sl_lvl = best - tr_d
            else:
                cur = ae
                if cur >= sl_lvl:
                    exit_px = (cur if cur > sl_lvl else sl_lvl) + slip
                    reason = 0
                elif tp_lvl > 0.0 and cur <= tp_lvl:
                    exit_px = tp_lvl
                    reason = 1
                elif t >= hold_end:
                    exit_px = cur + slip
                    reason = 2
                elif _tow(t) >= fclose:
                    exit_px = cur + slip
                    reason = 3
                elif cur < best:
                    best = cur
                    if tr_d > 0 and best + tr_d < sl_lvl:
                        sl_lvl = best + tr_d
            if reason >= 0:
                exit_ts = t
                break
            q += 1
        if reason < 0:
            ae_, be_ = _eff(ask[n - 1] / PTS, bid[n - 1] / PTS, prm[P_SPREADMULT])
            exit_px = (be_ - slip) if d > 0 else (ae_ + slip)
            reason = 4
            exit_ts = ts[n - 1]
        net_pips = (exit_px - fill) * d / 1e-4 - comm_pips
        out[c, 0] = 1.0
        out[c, 1] = fill
        out[c, 2] = exit_px
        out[c, 3] = exit_ts
        out[c, 4] = reason
        out[c, 5] = net_pips
        out[c, 6] = net_pips / (sl_d / 1e-4)
        out[c, 7] = (ask[m] - bid[m]) / PTS / 1e-4 * prm[P_SPREADMULT]
    return out
