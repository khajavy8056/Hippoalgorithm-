"""Tick-by-tick EURUSD backtest core (numba). Prices are stored as int 'points' (1e-5).
Conventions: ts = UTC epoch milliseconds. Long fills at ask, exits at bid; shorts the reverse.
"""
import numpy as np
from numba import njit

PTS = 100000.0            # price points per 1.0 (EURUSD: 5 decimals)
PIP = 1e-4                # 1 pip in price units
CONTRACT = 100000.0       # EURUSD standard lot in base currency
PIP_VALUE_LOT = 10.0      # USD per pip per standard lot (USD account)
MS_MIN = 60000
MS_HOUR = 3600000
MS_DAY = 86400000
MS_WEEK = 7 * MS_DAY

# engine parameter vector indices
P_EQ0 = 0        # starting equity (USD)
P_RISK = 1       # fraction of equity risked per trade
P_MAXLEV = 2     # max leverage (notional / equity)
P_COMM = 3       # commission USD per standard lot per side
P_SLIP = 4       # adverse slippage per fill, price units
P_LAT = 5        # latency between decision time and earliest fill (ms)
P_FILLWIN = 6    # max delay after latency for a fill (ms)
P_MAXSPREAD = 7  # skip entry if spread > this (price units); <=0 disables
P_NOENTRY_TOW = 8   # no new entries when time-of-week >= this (ms, 0=Mon 00:00 UTC)
P_FCLOSE_TOW = 9    # force-close open trades when time-of-week >= this
P_MINLOT = 10
P_LOTSTEP = 11
P_SPREADMULT = 12
P_NPARAM = 13

# trade record columns
TR_ENTRY_TS, TR_EXIT_TS, TR_DIR, TR_ENTRY, TR_EXIT, TR_LOTS, TR_GROSS, TR_COMM, TR_NET, TR_REASON, \
    TR_RISK, TR_SL, TR_TP, TR_MAE, TR_MFE, TR_SPREAD_IN, TR_SIG, TR_HOLD_MIN = range(18)
TR_NCOL = 18
EXIT_NAMES = {0: "SL", 1: "TP", 2: "TIME", 3: "WEEKEND", 4: "EOD"}


def default_params(equity=10_000.0, risk=0.005, max_lev=30.0, comm=3.5, slip_pips=0.1,
                   latency_ms=1000, fillwin_ms=30_000, max_spread_pips=2.0):
    p = np.zeros(P_NPARAM, dtype=np.float64)
    p[P_EQ0] = equity
    p[P_RISK] = risk
    p[P_MAXLEV] = max_lev
    p[P_COMM] = comm
    p[P_SLIP] = slip_pips * PIP
    p[P_LAT] = latency_ms
    p[P_FILLWIN] = fillwin_ms
    p[P_MAXSPREAD] = max_spread_pips * PIP if max_spread_pips and max_spread_pips > 0 else 0.0
    p[P_NOENTRY_TOW] = 4 * MS_DAY + 20 * MS_HOUR          # Fri 20:00 UTC
    p[P_FCLOSE_TOW] = 4 * MS_DAY + 20 * MS_HOUR + 30 * MS_MIN  # Fri 20:30 UTC
    p[P_MINLOT] = 0.01
    p[P_LOTSTEP] = 0.01
    p[P_SPREADMULT] = 1.0
    return p


@njit(cache=True)
def _bsearch_left(ts, x, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if ts[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _tow(t):
    # time of week in ms, 0 = Monday 00:00 UTC (1970-01-01 was a Thursday)
    return (t + 3 * MS_DAY) % MS_WEEK


@njit(cache=True)
def _eff(a, b, k):
    # scale the quoted spread around mid by k (cost stress test); k=1 leaves quotes unchanged
    if k == 1.0:
        return a, b
    mid = 0.5 * (a + b)
    return mid + k * (a - mid), mid - k * (mid - b)


@njit(cache=True)
def _entry_ok(ask_p, bid_p, t, prm):
    sp = ask_p - bid_p
    maxsp = prm[P_MAXSPREAD]
    if maxsp > 0 and sp > maxsp:
        return False
    if _tow(t) >= prm[P_NOENTRY_TOW]:
        return False
    return True


@njit(cache=True)
def _size_lots(bal, risk, sl_d, fill, maxlev, minlot, lstep):
    lots_r = bal * risk / (sl_d * CONTRACT)
    lots_l = maxlev * bal / (CONTRACT * fill)
    lots = lots_r if lots_r < lots_l else lots_l
    lots = np.floor(lots / lstep + 1e-9) * lstep
    if lots < minlot - 1e-12:
        return 0.0
    return lots


@njit(cache=True)
def run_engine(ts, bid, ask, s_ts, s_dir, s_sl, s_tp, s_hold, prm):
    """Event-driven tick engine. s_* are sorted by decision time s_ts (ms).
    s_sl / s_tp are distances in price units from the fill; s_tp<=0 means no TP;
    s_hold is the time-stop in ms (<=0 means none). Returns trades, equity events, stats."""
    n = ts.shape[0]
    ns = s_ts.shape[0]
    lat = prm[P_LAT]
    fw = prm[P_FILLWIN]
    slip = prm[P_SLIP]
    comm = prm[P_COMM]
    risk = prm[P_RISK]
    maxlev = prm[P_MAXLEV]
    fclose = prm[P_FCLOSE_TOW]
    minlot = prm[P_MINLOT]
    lstep = prm[P_LOTSTEP]
    bal = prm[P_EQ0]
    eq0 = bal
    peak = bal
    max_dd_abs = 0.0
    max_dd_rel = 0.0
    max_tr = ns + 1
    tr = np.zeros((max_tr, TR_NCOL))
    ntr = 0
    span_min = 0
    if n > 0:
        span_min = (ts[n - 1] - ts[0]) // MS_MIN + 1
    max_eq = span_min + 2 * max_tr + 10
    eq = np.zeros((max_eq, 2))
    neq = 0
    last_min = -1
    if n == 0 or ns == 0:
        out_eq = np.zeros((1, 2))
        out_eq[0, 0] = 0.0
        out_eq[0, 1] = eq0
        return tr[:0], out_eq[:1], bal, max_dd_abs, max_dd_rel, 0
    eq[0, 0] = ts[0]
    eq[0, 1] = bal
    neq = 1
    j = 0
    i = 0
    while True:
        if j >= ns or bal <= 0.0 or i >= n:
            break
        t_ready = s_ts[j] + lat
        k = _bsearch_left(ts, t_ready, i, n)
        if k >= n:
            break
        if ts[k] - t_ready > fw:
            j += 1  # stale signal
            continue
        filled = False
        m = k
        fill = 0.0
        while m < n and ts[m] - t_ready <= fw:
            a, b = _eff(ask[m] / PTS, bid[m] / PTS, prm[P_SPREADMULT])
            if _entry_ok(a, b, ts[m], prm):
                d = s_dir[j]
                fill = a + slip if d > 0 else b - slip
                sl_d = s_sl[j]
                if sl_d > 0:
                    lots = _size_lots(bal, risk, sl_d, fill, maxlev, minlot, lstep)
                    if lots > 0:
                        filled = True
                        break
            m += 1
        j += 1
        if not filled:
            continue
        # ---- open position
        d = s_dir[j - 1]
        sl_d = s_sl[j - 1]
        tp_d = s_tp[j - 1]
        hold = s_hold[j - 1]
        sl_lvl = fill - sl_d if d > 0 else fill + sl_d
        tp_lvl = (fill + tp_d if d > 0 else fill - tp_d) if tp_d > 0 else 0.0
        hold_end = (ts[m] + hold) if hold > 0 else (1 << 62)
        comm_in = comm * lots
        bal -= comm_in
        entry_ts = ts[m]
        entry_idx = m
        mae = 0.0
        mfe = 0.0
        spread_in = (ask[m] - bid[m]) / PTS
        risk_usd = risk * (bal + comm_in)
        sig_idx = j - 1
        q = m + 1
        exit_px = 0.0
        reason = -1
        exit_ts = 0
        last_mtm = bal
        while q < n:
            t = ts[q]
            ae, be = _eff(ask[q] / PTS, bid[q] / PTS, prm[P_SPREADMULT])
            if d > 0:
                cur = be
                exc = cur - fill
                mtm = bal + exc * lots * CONTRACT
                if exc < mae:
                    mae = exc
                if exc > mfe:
                    mfe = exc
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
            else:
                cur = ae
                exc = fill - cur
                mtm = bal + exc * lots * CONTRACT
                if exc < mae:
                    mae = exc
                if exc > mfe:
                    mfe = exc
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
            if reason >= 0:
                exit_ts = t
                break
            last_mtm = mtm
            if mtm > peak:
                peak = mtm
            dd = peak - mtm
            if dd > max_dd_abs:
                max_dd_abs = dd
            if peak > 0 and dd / peak > max_dd_rel:
                max_dd_rel = dd / peak
            mn = t // MS_MIN
            if mn != last_min:
                last_min = mn
                if neq < max_eq:
                    eq[neq, 0] = t
                    eq[neq, 1] = mtm
                    neq += 1
            q += 1
        if reason < 0:
            # end of data: close at last available price
            t = ts[n - 1]
            cur_a, cur_b = _eff(ask[n - 1] / PTS, bid[n - 1] / PTS, prm[P_SPREADMULT])
            exit_px = cur_b - slip if d > 0 else cur_a + slip
            reason = 4
            exit_ts = t
            q = n - 1
        gross = (exit_px - fill) * d * lots * CONTRACT
        comm_out = comm * lots
        bal = bal + gross - comm_out
        net = gross - comm_in - comm_out
        if ntr < max_tr:
            tr[ntr, TR_ENTRY_TS] = entry_ts
            tr[ntr, TR_EXIT_TS] = exit_ts
            tr[ntr, TR_DIR] = d
            tr[ntr, TR_ENTRY] = fill
            tr[ntr, TR_EXIT] = exit_px
            tr[ntr, TR_LOTS] = lots
            tr[ntr, TR_GROSS] = gross
            tr[ntr, TR_COMM] = comm_in + comm_out
            tr[ntr, TR_NET] = net
            tr[ntr, TR_REASON] = reason
            tr[ntr, TR_RISK] = risk_usd
            tr[ntr, TR_SL] = sl_d
            tr[ntr, TR_TP] = tp_d
            tr[ntr, TR_MAE] = mae
            tr[ntr, TR_MFE] = mfe
            tr[ntr, TR_SPREAD_IN] = spread_in
            tr[ntr, TR_SIG] = sig_idx
            tr[ntr, TR_HOLD_MIN] = (exit_ts - entry_ts) / MS_MIN
            ntr += 1
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_dd_abs:
            max_dd_abs = dd
        if peak > 0 and dd / peak > max_dd_rel:
            max_dd_rel = dd / peak
        if neq < max_eq:
            eq[neq, 0] = exit_ts
            eq[neq, 1] = bal
            neq += 1
        i = q + 1
        # signals decided while in a trade are discarded
        while j < ns and s_ts[j] <= exit_ts:
            j += 1
        if reason == 4:
            break
    return tr[:ntr], eq[:neq], bal, max_dd_abs, max_dd_rel, ntr


@njit(cache=True)
def candidate_outcomes(ts, bid, ask, c_ts, c_dir, c_sl, c_tp, c_hold, prm):
    """Stateless tick-level outcome of each candidate (independent of portfolio state).
    Returns array [filled, entry, exit, exit_ts, reason, net_pips, R, spread_in_pips]."""
    n = ts.shape[0]
    nc = c_ts.shape[0]
    lat = prm[P_LAT]
    fw = prm[P_FILLWIN]
    slip = prm[P_SLIP]
    comm_pips = 2.0 * prm[P_COMM] / PIP_VALUE_LOT  # round-trip commission per lot in pips
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
        sl_lvl = fill - sl_d if d > 0 else fill + sl_d
        tp_lvl = (fill + tp_d if d > 0 else fill - tp_d) if tp_d > 0 else 0.0
        hold_end = (ts[m] + hold) if hold > 0 else (1 << 62)
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
            if reason >= 0:
                exit_ts = t
                break
            q += 1
        if reason < 0:
            ae_, be_ = _eff(ask[n - 1] / PTS, bid[n - 1] / PTS, prm[P_SPREADMULT])
            exit_px = (be_ - slip) if d > 0 else (ae_ + slip)
            reason = 4
            exit_ts = ts[n - 1]
        net_pips = (exit_px - fill) * d / PIP - comm_pips
        out[c, 0] = 1.0
        out[c, 1] = fill
        out[c, 2] = exit_px
        out[c, 3] = exit_ts
        out[c, 4] = reason
        out[c, 5] = net_pips
        out[c, 6] = net_pips / (sl_d / PIP)
        out[c, 7] = (ask[m] - bid[m]) / PTS / PIP * prm[P_SPREADMULT]
    return out


@njit(cache=True)
def build_m1(ts, bid, ask, t0_min, n_min):
    """One pass over ticks -> minute bars of MID with tick statistics."""
    o = np.full(n_min, np.nan)
    h = np.full(n_min, np.nan)
    l = np.full(n_min, np.nan)
    c = np.full(n_min, np.nan)
    cnt = np.zeros(n_min, np.int64)
    sp_sum = np.zeros(n_min)
    sp_max = np.zeros(n_min)
    up = np.zeros(n_min, np.int64)
    dn = np.zeros(n_min, np.int64)
    prev_mid = np.nan
    for i in range(ts.shape[0]):
        k = ts[i] // MS_MIN - t0_min
        if k < 0 or k >= n_min:
            continue
        mid = 0.5 * (bid[i] + ask[i]) / PTS
        sp = (ask[i] - bid[i]) / PTS
        if np.isnan(o[k]):
            o[k] = mid
            h[k] = mid
            l[k] = mid
        if mid > h[k]:
            h[k] = mid
        if mid < l[k]:
            l[k] = mid
        c[k] = mid
        cnt[k] += 1
        sp_sum[k] += sp
        if sp > sp_max[k]:
            sp_max[k] = sp
        if not np.isnan(prev_mid):
            if mid > prev_mid:
                up[k] += 1
            elif mid < prev_mid:
                dn[k] += 1
        prev_mid = mid
    return o, h, l, c, cnt, sp_sum, sp_max, up, dn
