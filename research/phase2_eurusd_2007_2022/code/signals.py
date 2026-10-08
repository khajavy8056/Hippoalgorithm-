import numpy as np, pandas as pd
from numba import njit
from core import build_m1, MS_MIN, PTS, PIP, MS_DAY, MS_HOUR

EPOCH = pd.Timestamp("1970-01-01")


def to_ms(idx):
    return np.asarray((pd.DatetimeIndex(idx) - EPOCH) // pd.Timedelta(milliseconds=1), dtype=np.int64)


def build_m1_frame(ts, bid, ask):
    t0min = int(ts[0] // MS_MIN)
    nmin = int(ts[-1] // MS_MIN - t0min + 1)
    o, h, l, c, cnt, sp_sum, sp_max, up, dn = build_m1(ts, bid, ask, t0min, nmin)
    idx = pd.to_datetime(t0min * MS_MIN + np.arange(nmin, dtype=np.int64) * MS_MIN, unit="ms")
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "ticks": cnt,
                       "sp_sum": sp_sum, "sp_max": sp_max, "up": up, "dn": dn}, index=idx)
    m1 = m1[m1["ticks"] > 0].copy()
    m1["spread"] = m1["sp_sum"] / m1["ticks"] / PIP
    return m1


def resample_bars(m1, rule):
    g = m1.resample(rule, label="left", closed="left")
    b = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "ticks": g["ticks"].sum(), "sp_sum": g["sp_sum"].sum(), "sp_max": g["sp_max"].max(),
        "up": g["up"].sum(), "dn": g["dn"].sum()})
    b = b[b["ticks"] > 0].copy()
    b["spread"] = b["sp_sum"] / b["ticks"] / PIP
    b["dec_ms"] = to_ms(b.index + pd.Timedelta(rule))  # decision time = bar close
    return b


def atr(b, n=14):
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(b, n=14):
    up = b["high"].diff()
    dn = -b["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(b, n)
    pdi = 100 * pd.Series(pdm, index=b.index).ewm(alpha=1.0 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm, index=b.index).ewm(alpha=1.0 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean()


def _cands(dec_ms, dirn, sl, tp, hold_ms, mask):
    df = pd.DataFrame({"dec_ms": dec_ms, "dir": dirn, "sl": sl, "tp": tp, "hold": hold_ms})
    df = df[mask & np.isfinite(df["sl"]) & (df["sl"] > 0)]
    return df.sort_values("dec_ms", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------------
# F1: Asia-range breakout at London open (M5 bars)
# ----------------------------------------------------------------------------------
def f1_asia_breakout(b5, range_end_h=7, win_end_h=10, buf_pips=0.0, sl_mode="range", atr_mult=1.0,
                     rr=1.5, rmin_pips=12, rmax_pips=80, min_sl_pips=6, hold_h=8):
    d = b5.copy()
    d["date"] = d.index.normalize()
    hr = d.index.hour + d.index.minute / 60
    asia = d[hr < range_end_h].groupby("date").agg(hi=("high", "max"), lo=("low", "min"))
    d = d.join(asia, on="date")
    rng = (d["hi"] - d["lo"]) / PIP
    in_win = (hr >= range_end_h) & (hr < win_end_h)
    buf = buf_pips * PIP
    A = d["atr"] if "atr" in d else atr(b5)
    long_b = in_win & (d["close"] > d["hi"] + buf) & rng.between(rmin_pips, rmax_pips)
    short_b = in_win & (d["close"] < d["lo"] - buf) & rng.between(rmin_pips, rmax_pips)
    trig = (long_b | short_b).astype(bool)
    # first breakout per day only
    first = trig & (trig.groupby(d["date"]).cumsum() == 1)
    dirn = np.where(long_b, 1, -1)
    if sl_mode == "range":
        sl = np.where(dirn > 0, d["close"] - d["lo"], d["hi"] - d["close"]) + buf
    else:
        sl = atr_mult * A.values
    sl = np.maximum(sl, min_sl_pips * PIP)
    tp = rr * sl
    hold = hold_h * MS_HOUR
    m = first.values
    return _cands(d["dec_ms"].values, dirn, sl, tp, np.full(len(d), hold), m)


# ----------------------------------------------------------------------------------
# F2: Donchian breakout on H1 with EMA200 trend filter
# ----------------------------------------------------------------------------------
def f2_donchian(b1, n=48, k=2.0, rr=2.0, ema_n=200, hold_h=120):
    A = atr(b1)
    up = b1["high"].rolling(n).max().shift(1)
    lo = b1["low"].rolling(n).min().shift(1)
    ema = b1["close"].ewm(span=ema_n, adjust=False).mean()
    lb = (b1["close"] > up) & (b1["close"] > ema)
    sb = (b1["close"] < lo) & (b1["close"] < ema)
    lb_first = lb & ~lb.shift(1, fill_value=False)
    sb_first = sb & ~sb.shift(1, fill_value=False)
    trig = (lb_first | sb_first).values
    dirn = np.where(lb_first.values, 1, -1)
    sl = k * A.values
    tp = rr * sl
    return _cands(b1["dec_ms"].values, dirn, sl, tp, np.full(len(b1), hold_h * MS_HOUR), trig)


# ----------------------------------------------------------------------------------
# F3: Mean reversion on M15 (Bollinger + RSI + low-ADX regime)
# ----------------------------------------------------------------------------------
def f3_mean_reversion(b15, bb_k=2.0, sl_atr=1.5, rsi_lo=30, rsi_hi=70, adx_max=25, hold_bars=8):
    A = atr(b15)
    mid = b15["close"].rolling(20).mean()
    sd = b15["close"].rolling(20).std()
    lower = mid - bb_k * sd
    upper = mid + bb_k * sd
    R = rsi(b15["close"])
    X = adx(b15)
    lb = (b15["close"] < lower) & (R < rsi_lo) & (X < adx_max)
    sb = (b15["close"] > upper) & (R > rsi_hi) & (X < adx_max)
    lb_first = lb & ~lb.shift(1, fill_value=False)
    sb_first = sb & ~sb.shift(1, fill_value=False)
    trig = (lb_first | sb_first).values
    dirn = np.where(lb_first.values, 1, -1)
    sl = sl_atr * A.values
    tp = np.abs(mid - b15["close"]).values
    tp = np.where(tp > 0.5 * sl, tp, 0.5 * sl)  # guard tiny targets
    return _cands(b15["dec_ms"].values, dirn, sl, tp, np.full(len(b15), hold_bars * 15 * MS_MIN), trig)


# ----------------------------------------------------------------------------------
# F4: Pivot impulse (AB) + 50-78.6% retracement + reversal close (causal, M15)
# simplified re-implementation of the 'pivot settlement' idea; all pivots confirmed
# only after n bars (no look-ahead)
# ----------------------------------------------------------------------------------
@njit(cache=True)
def _pivot_scan(high, low, opn, close, atr_v, nf, ab_atr, min_b, max_b, rr, buf_atr, max_wait, max_sl_atr):
    N = high.shape[0]
    sig_i = np.zeros(N, np.int64)
    sig_d = np.zeros(N, np.int64)
    sig_sl = np.zeros(N)
    sig_tp = np.zeros(N)
    ns = 0
    last_lo_i = -1
    last_lo_p = np.nan
    last_hi_i = -1
    last_hi_p = np.nan
    last_type = 0  # 1 = last confirmed pivot is high, -1 = low
    act = 0
    A_p = np.nan
    B_p = np.nan
    B_i = -1
    AB = np.nan
    C_p = np.nan
    for c in range(nf, N):
        p = c - nf
        ph = True
        pl = True
        for q in range(p - nf, p + nf + 1):
            if q < 0 or q >= N or q == p:
                continue
            if high[q] >= high[p]:
                ph = False
            if low[q] <= low[p]:
                pl = False
        if p - nf < 0:
            ph = False
            pl = False
        # ---- pivot confirmations at bar c
        if ph:
            if last_type == -1 and last_lo_i >= 0:
                size = high[p] - last_lo_p
                span = p - last_lo_i
                if size >= ab_atr * atr_v[c] and span >= min_b and span <= max_b:
                    act = 1
                    A_p = last_lo_p
                    B_p = high[p]
                    B_i = p
                    AB = size
                    C_p = low[p]
                else:
                    act = 0
            last_hi_i = p
            last_hi_p = high[p]
            last_type = 1
        if pl:
            if last_type == 1 and last_hi_i >= 0:
                size = last_hi_p - low[p]
                span = p - last_hi_i
                if size >= ab_atr * atr_v[c] and span >= min_b and span <= max_b:
                    act = -1
                    A_p = last_hi_p
                    B_p = low[p]
                    B_i = p
                    AB = size
                    C_p = high[p]
                else:
                    act = 0
            last_lo_i = p
            last_lo_p = low[p]
            last_type = -1
        # ---- manage active impulse on bar c (only bars after B)
        if act != 0 and c > B_i:
            if c - B_i > max_wait:
                act = 0
                continue
            if act == 1:
                if low[c] < A_p:
                    act = 0
                    continue
                if low[c] < C_p:
                    C_p = low[c]
                lvl50 = B_p - 0.5 * AB
                lvl786 = B_p - 0.786 * AB
                if C_p <= lvl50 and C_p >= lvl786 and close[c] > opn[c] and close[c] >= lvl50:
                    sl_d = close[c] - (C_p - buf_atr * atr_v[c])
                    if sl_d > 0 and sl_d <= max_sl_atr * atr_v[c]:
                        sig_i[ns] = c
                        sig_d[ns] = 1
                        sig_sl[ns] = sl_d
                        sig_tp[ns] = rr * sl_d
                        ns += 1
                    act = 0
            else:
                if high[c] > A_p:
                    act = 0
                    continue
                if high[c] > C_p:
                    C_p = high[c]
                lvl50 = B_p + 0.5 * AB
                lvl786 = B_p + 0.786 * AB
                if C_p >= lvl50 and C_p <= lvl786 and close[c] < opn[c] and close[c] <= lvl50:
                    sl_d = (C_p + buf_atr * atr_v[c]) - close[c]
                    if sl_d > 0 and sl_d <= max_sl_atr * atr_v[c]:
                        sig_i[ns] = c
                        sig_d[ns] = -1
                        sig_sl[ns] = sl_d
                        sig_tp[ns] = rr * sl_d
                        ns += 1
                    act = 0
    return sig_i[:ns], sig_d[:ns], sig_sl[:ns], sig_tp[:ns]


def f4_pivot_retrace(b15, ab_atr=2.0, rr=1.5, nf=3, min_b=5, max_b=60, max_wait=40, buf_atr=0.2, max_sl_atr=4.0):
    A = atr(b15).values
    idx, d, sl, tp = _pivot_scan(b15["high"].values.astype(np.float64), b15["low"].values.astype(np.float64),
                                 b15["open"].values.astype(np.float64), b15["close"].values.astype(np.float64),
                                 np.nan_to_num(A, nan=1e-4), nf, ab_atr, min_b, max_b, rr, buf_atr, max_wait, max_sl_atr)
    dec = b15["dec_ms"].values[idx]
    return pd.DataFrame({"dec_ms": dec, "dir": d, "sl": sl, "tp": tp, "hold": np.full(len(idx), 48 * 15 * MS_MIN)})
