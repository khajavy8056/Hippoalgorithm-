"""HIPO 'Pivot-Settlement' (ستاپ پیوت تسویه) rule engine — causal state machine on OHLC bars.

Orientation: the scanner treats the setup as a SHORT setup (AB is an up-leg, BC a pull-back,
CD a weaker up-leg, the signal candle breaks B, entry after a body close below the signal low).
LONG setups are the mirror image: the same scanner is run on negated prices.
All thresholds are ATR-relative, so the rules are scale-free (index CFD or FX).
"""
import numpy as np
from numba import njit

# parameter vector indices
P_AB_MIN_BARS, P_AB_MAX_NOISE, P_AB_LONG_LO, P_AB_LONG_HI, P_AB_THREE_MIN, P_AB2_BAR_ATR = 0, 1, 2, 3, 4, 5
P_BC_RET_LO, P_BC_RET_HI, P_BC_MIN_BARS, P_CD_START_ATR = 6, 7, 8, 9
P_SIG_MIN_RANGE, P_SIG_POS_LO, P_SIG_POS_HI, P_SIG_MAX_WICK, P_STRICT_BOX = 10, 11, 12, 13, 14
P_CONF_BODY_ATR, P_CONF_MAX_WAIT, P_SL_BUF_ATR, P_TP_R = 15, 16, 17, 18
N_PARAMS = 19

DEFAULTS = dict(
    AB_MIN_BARS=3, AB_MAX_NOISE=1, AB_LONG_LO=2.0, AB_LONG_HI=5.0, AB_THREE_MIN=1.5, AB2_BAR_ATR=1.0,
    BC_RET_LO=0.20, BC_RET_HI=0.50, BC_MIN_BARS=3, CD_START_ATR=0.5,
    SIG_MIN_RANGE=0.8, SIG_POS_LO=0.30, SIG_POS_HI=0.70, SIG_MAX_WICK=0.35, STRICT_BOX=1.0,
    CONF_BODY_ATR=0.5, CONF_MAX_WAIT=15, SL_BUF_ATR=0.25, TP_R=2.5,
)
_ORDER = ["AB_MIN_BARS", "AB_MAX_NOISE", "AB_LONG_LO", "AB_LONG_HI", "AB_THREE_MIN", "AB2_BAR_ATR",
          "BC_RET_LO", "BC_RET_HI", "BC_MIN_BARS", "CD_START_ATR",
          "SIG_MIN_RANGE", "SIG_POS_LO", "SIG_POS_HI", "SIG_MAX_WICK", "STRICT_BOX",
          "CONF_BODY_ATR", "CONF_MAX_WAIT", "SL_BUF_ATR", "TP_R"]

def params_vector(**kw):
    d = dict(DEFAULTS); d.update(kw)
    return np.array([float(d[k]) for k in _ORDER], dtype=np.float64)

# output columns of scan_short
OUT_COLS = ["entry_idx", "sig_idx", "a_idx", "b_idx", "c_idx", "entry", "sl", "tp",
            "ab_size", "atr", "minor_hi", "bc_bars", "ab_bars", "time_div"]
NOUT = len(OUT_COLS)

@njit(cache=True)
def _ab_valid(O, H, L, C, a, b, ab, atr, P):
    cnt = b - a + 1
    if cnt < 2:
        return False
    if cnt == 2:
        # PDF: two-candle AB is accepted (with discount) when each candle is larger than ATR
        if not (C[a] >= O[a] and C[b] >= O[b]):
            return False
        return (H[a] - L[a] > P[P_AB2_BAR_ATR] * atr) and (H[b] - L[b] > P[P_AB2_BAR_ATR] * atr)
    bull = 0; noise = 0
    for k in range(a, b + 1):
        if C[k] >= O[k]:
            bull += 1
        else:
            noise += 1
    if bull < int(P[P_AB_MIN_BARS]) or noise > int(P[P_AB_MAX_NOISE]):
        return False
    if cnt == 3:
        return ab >= P[P_AB_THREE_MIN] * atr
    # more than three candles: AB must be 200%-500% of ATR (PDF)
    return (ab >= P[P_AB_LONG_LO] * atr) and (ab <= P[P_AB_LONG_HI] * atr)

@njit(cache=True)
def scan_short(O, H, L, C, ATR, P):
    n = C.shape[0]
    out = np.zeros((n // 3 + 10, NOUT))
    k_out = 0
    phase = 0
    a = 0; la = L[0]; b = 0; hb = H[0]
    mlow = np.inf; mh = -np.inf
    c = -1; lc = np.inf; box_top = 0.0
    bodymin = np.inf
    s = -1; ls = 0.0; hs = 0.0; minor = -np.inf; wait = 0
    ab = 0.0
    i = 1
    while i < n:
        atr = ATR[i]
        if not (atr > 0.0):
            i += 1
            continue
        if phase == 0:
            if L[i] < la:                       # new lower low -> new candidate A
                a = i; la = L[i]; b = i; hb = H[i]; mlow = np.inf; mh = -np.inf
                i += 1; continue
            if H[i] > hb:                       # higher high -> candidate B moves up
                b = i; hb = H[i]; mlow = np.inf; mh = -np.inf
                i += 1; continue
            mlow = min(mlow, L[i])              # bar after B
            ab = hb - la
            if ab > 0.0 and (hb - mlow) >= P[P_BC_RET_LO] * ab and (b - a + 1) >= 2:
                if _ab_valid(O, H, L, C, a, b, ab, ATR[b], P):
                    phase = 1; c = -1; lc = np.inf; bodymin = np.inf
                    mh = -np.inf
                    # re-process this bar in phase 1 (it is the first BC candle)
                    continue
                elif (b - a + 1) >= 3:   # complete (>=3 candles) but invalid AB: rewind to bar after A
                    a = a + 1; la = L[a]; b = a; hb = H[a]; mlow = np.inf; mh = -np.inf
                    i = a + 1; continue
                # two-candle AB not yet valid: keep scanning (the impulse may still extend B)
            i += 1
            continue

        if phase == 1:  # ---------------- BC: pull-back from B to C
            if H[i] > hb:                       # B not the top any more -> extend AB
                b = i; hb = H[i]; phase = 0; mlow = np.inf; mh = -np.inf
                i += 1; continue
            if L[i] <= la:                      # BC touches A -> setup fails: rewind to bar after A
                a = a + 1; la = L[a]; b = a; hb = H[a]; phase = 0; mlow = np.inf; mh = -np.inf
                i = a + 1; continue
            bl = min(O[i], C[i])
            if bl < hb - P[P_BC_RET_HI] * ab:   # body closes beyond 50% of AB -> fail: rewind
                a = a + 1; la = L[a]; b = a; hb = H[a]; phase = 0; mlow = np.inf; mh = -np.inf
                i = a + 1; continue
            if L[i] < lc:                       # this bar is the lowest low so far -> C candidate
                lc = L[i]; c = i
            if i > c and H[i] >= lc + P[P_CD_START_ATR] * atr:   # CD starts (rally off the BC low)
                bc_bars = c - b
                ret_body = (hb - bodymin) / ab
                if bc_bars >= int(P[P_BC_MIN_BARS]) and ret_body >= P[P_BC_RET_LO] and ret_body <= P[P_BC_RET_HI]:
                    phase = 2; box_top = lc + ab; s = -1; minor = -np.inf; wait = 0
                    continue                    # re-process this bar in phase 2
                elif bc_bars < int(P[P_BC_MIN_BARS]):
                    pass                        # too early: the bounce may still be part of BC -> keep scanning
                else:   # BC/CD check failed: rewind to the bar after A
                    a = a + 1; la = L[a]; b = a; hb = H[a]; phase = 0; mlow = np.inf; mh = -np.inf
                    i = a + 1; continue
            bodymin = min(bodymin, bl)
            mh = max(mh, H[i])
            i += 1
            continue

        # ---------------- phase 2: CD, signal candle, confirmation candle
        if L[i] < lc and s < 0:            # C extended lower before any signal: back to BC phase
            phase = 1; c = i; lc = L[i]
            bodymin = min(bodymin, min(O[i], C[i])); mh = max(mh, H[i])
            i += 1; continue
        fail = False
        if (H[i] >= box_top and P[P_STRICT_BOX] > 0.5) or C[i] >= box_top:
            fail = True                         # CD >= AB (price divergence violated) or close beyond box
        if L[i] < lc:
            fail = True                         # C broken -> structure invalid
        if fail:
            # restart a new AB candidate from C (the pivot low), rescan from c+1
            a = c; la = lc; b = c; hb = H[c]; phase = 0; mlow = np.inf; mh = -np.inf
            i = c + 1
            continue

        if s < 0:
            if H[i] > hb:
                body_top = max(O[i], C[i])
                hh = (H[i] > mh) or (body_top > mh)          # higher high vs candles between B and i
                rng = H[i] - L[i]
                if hh and rng >= P[P_SIG_MIN_RANGE] * atr and C[i] < box_top:
                    pos = (hb - L[i]) / rng                    # B should sit near the middle of the candle
                    if pos >= P[P_SIG_POS_LO] and pos <= P[P_SIG_POS_HI]:
                        upper = H[i] - body_top
                        if upper <= P[P_SIG_MAX_WICK] * rng:   # no large upper wick (selling pressure)
                            s = i; ls = L[i]; hs = H[i]; minor = -np.inf; wait = 0
            mh = max(mh, H[i])
            i += 1
            continue

        # waiting for confirmation (bars after the signal candle)
        wait += 1
        if C[i] > hs:
            minor = max(minor, H[i])                   # a candle closed above the signal candle
        if (O[i] - C[i]) >= P[P_CONF_BODY_ATR] * atr and C[i] < ls and C[i] < O[i]:
            sl = max(hs, minor) + P[P_SL_BUF_ATR] * atr
            entry = C[i]
            R = sl - entry
            if R > 0.0:
                tp = entry - P[P_TP_R] * R
                if k_out < out.shape[0]:
                    out[k_out, 0] = i; out[k_out, 1] = s; out[k_out, 2] = a; out[k_out, 3] = b
                    out[k_out, 4] = c; out[k_out, 5] = entry; out[k_out, 6] = sl; out[k_out, 7] = tp
                    out[k_out, 8] = ab; out[k_out, 9] = atr
                    out[k_out, 10] = 1.0 if minor > -np.inf else 0.0
                    out[k_out, 11] = c - b; out[k_out, 12] = b - a + 1
                    out[k_out, 13] = 1.0 if (c - b) > (b - a + 1) else 0.0
                k_out += 1
            # one setup at a time: restart the AB search after the signal bar
            phase = 0; a = i; la = L[i]; b = i; hb = H[i]; mlow = np.inf; mh = -np.inf
            i += 1
            continue
        if C[i] >= box_top or wait > P[P_CONF_MAX_WAIT]:
            a = c; la = lc; b = c; hb = H[c]; phase = 0; mlow = np.inf; mh = -np.inf
            i = c + 1
            continue
        i += 1
    return out[:k_out]

def scan_direction(O, H, L, C, ATR, P, direction):
    """direction=+1 -> short setups (AB up); -1 -> long setups (mirror). Returns (k, NOUT) array."""
    if direction > 0:
        res = scan_short(O, H, L, C, ATR, P)
        return res
    res = scan_short(-O, -L, -H, -C, ATR, P)     # mirror: negate and swap high/low
    if len(res):
        res = res.copy()
        for col in (5, 6, 7):
            res[:, col] = -res[:, col]
    return res

def atr_wilder(H, L, C, n=14):
    H = np.asarray(H, float); L = np.asarray(L, float); C = np.asarray(C, float)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev_c), np.abs(L - prev_c)))
    out = np.full(len(C), np.nan)
    if len(C) <= n:
        return out
    out[n] = tr[1:n+1].mean()
    a = 1.0 / n
    for i in range(n + 1, len(C)):
        out[i] = out[i-1] + a * (tr[i] - out[i-1])
    return out
