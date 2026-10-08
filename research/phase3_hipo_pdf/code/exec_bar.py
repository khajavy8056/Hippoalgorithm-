"""Bar-level (M1 bid/ask) execution of HIPO signals. One position at a time, risk-based sizing."""
import numpy as np
from numba import njit

PIP = 1e-4
UNITS_PER_LOT = 100_000.0
MS_DAY = 86_400_000
MS_MIN = 60_000

# executor parameter vector
E_EQ0, E_RISK, E_MAXLEV, E_COMM, E_SLIP, E_MAXSPREAD, E_MAXHOLD, E_NOENTRY_TOW, E_FCLOSE_TOW, E_MINLOT = range(10)
def exec_params(equity=10_000.0, risk=0.005, maxlev=30.0, comm=3.5, slip_pips=0.1, max_spread_pips=2.0,
                max_hold_min=1440, minlot=0.01):
    p = np.zeros(10)
    p[E_EQ0] = equity; p[E_RISK] = risk; p[E_MAXLEV] = maxlev; p[E_COMM] = comm
    p[E_SLIP] = slip_pips * PIP; p[E_MAXSPREAD] = max_spread_pips * PIP
    p[E_MAXHOLD] = max_hold_min
    p[E_NOENTRY_TOW] = 4 * MS_DAY + 20 * 3_600_000          # Fri 20:00 UTC
    p[E_FCLOSE_TOW] = 4 * MS_DAY + 20 * 3_600_000 + 30 * MS_MIN  # Fri 20:30 UTC
    p[E_MINLOT] = minlot
    return p

TR_COLS = ["entry_k", "exit_k", "dir", "fill", "exit_px", "sl", "tp", "lots", "pnl", "R", "reason", "equity_after"]
REASON = {0: "SL", 1: "TP", 2: "TIME", 3: "WEEKEND"}

@njit(cache=True)
def _tow_ms(key_min):
    ms = key_min * 60000
    d = ms // 86400000
    wd = (d + 3) % 7
    return wd * 86400000 + (ms % 86400000)

@njit(cache=True)
def exec_bars(key, bo, bh, bl, bc, ao, ah, al, ac, s_k, s_dir, s_entry, s_sl, s_tp, prm):
    """s_k: entry bar index (sorted), s_dir: +1 long / -1 short, s_entry/s_sl/s_tp: MID prices."""
    n = key.shape[0]
    ns = s_k.shape[0]
    out = np.full((ns, 12), np.nan)
    eq = prm[E_EQ0]
    risk = prm[E_RISK]; maxlev = prm[E_MAXLEV]; comm = prm[E_COMM]; slip = prm[E_SLIP]
    maxsp = prm[E_MAXSPREAD]; maxhold = int(prm[E_MAXHOLD]); fclose = prm[E_FCLOSE_TOW]
    noentry = prm[E_NOENTRY_TOW]; minlot = prm[E_MINLOT]
    busy_until = -1
    nt = 0
    for q in range(ns):
        k = s_k[q]
        if k <= busy_until or k >= n - 1:
            continue
        d = s_dir[q]
        hs = 0.5 * (ac[k] - bc[k])                 # half spread at entry bar
        if (ac[k] - bc[k]) > maxsp:
            continue
        if _tow_ms(key[k]) >= noentry:
            continue
        if d < 0:                                  # SHORT: sell at bid, stop/target are buy-side (ask)
            fill = bc[k] - slip
            sl_t = s_sl[q] + hs; tp_t = s_tp[q] + hs
            R = sl_t - fill
        else:                                      # LONG: buy at ask
            fill = ac[k] + slip
            sl_t = s_sl[q] - hs; tp_t = s_tp[q] - hs
            R = fill - sl_t
        if not (R > 0.0):
            continue
        lots = np.floor(risk * eq / (R * UNITS_PER_LOT) / minlot) * minlot
        if lots < minlot:
            lots = minlot
        if lots * UNITS_PER_LOT * fill > maxlev * eq:      # leverage cap
            lots = np.floor(maxlev * eq / (UNITS_PER_LOT * fill) / minlot) * minlot
            if lots < minlot:
                continue
        # simulate on subsequent bars
        exit_j = -1; exit_px = np.nan; reason = -1
        j = k + 1
        while j < n:
            if _tow_ms(key[j]) >= fclose and key[j] - key[k] >= 0:
                # forced weekend close at bar close
                exit_px = (ac[j] + slip) if d < 0 else (bc[j] - slip)
                exit_j = j; reason = 3; break
            if j - k >= maxhold:
                exit_px = (ac[j] + slip) if d < 0 else (bc[j] - slip)
                exit_j = j; reason = 2; break
            if d < 0:
                sl_hit = ah[j] >= sl_t
                tp_hit = al[j] <= tp_t
                if sl_hit:
                    exit_px = sl_t + slip; exit_j = j; reason = 0; break     # SL first if both (conservative)
                if tp_hit:
                    exit_px = tp_t; exit_j = j; reason = 1; break
            else:
                sl_hit = bl[j] <= sl_t
                tp_hit = bh[j] >= tp_t
                if sl_hit:
                    exit_px = sl_t - slip; exit_j = j; reason = 0; break
                if tp_hit:
                    exit_px = tp_t; exit_j = j; reason = 1; break
            j += 1
        if exit_j < 0:
            break                                  # data ended with an open position
        if d < 0:
            gross = (fill - exit_px) * UNITS_PER_LOT * lots
        else:
            gross = (exit_px - fill) * UNITS_PER_LOT * lots
        cost = 2.0 * comm * lots
        pnl = gross - cost
        eq = eq + pnl
        Rreal = pnl / (risk * (eq - pnl) if False else (risk * (eq - pnl)))
        out[nt, 0] = k; out[nt, 1] = exit_j; out[nt, 2] = d; out[nt, 3] = fill; out[nt, 4] = exit_px
        out[nt, 5] = sl_t; out[nt, 6] = tp_t; out[nt, 7] = lots; out[nt, 8] = pnl; out[nt, 9] = Rreal
        out[nt, 10] = reason; out[nt, 11] = eq
        nt += 1
        busy_until = exit_j
    return out[:nt]

def mid_bars(bo, bh, bl, bc, ao, ah, al, ac):
    return ((bo + ao) / 2, (bh + ah) / 2, (bl + al) / 2, (bc + ac) / 2)
