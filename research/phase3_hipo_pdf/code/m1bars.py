import numpy as np, glob, os
from numba import njit

MS_MIN = 60_000
MS_DAY = 86_400_000
PRICE_SCALE = 1e5          # stored bid/ask are integers scaled by 1e5 (EURUSD 5-decimal)

@njit(cache=True)
def m1_bars_bidask(ts, bid, ask, scale):
    """Minute OHLC of bid and ask from sorted ticks. Returns arrays trimmed to the bars that exist."""
    n = ts.shape[0]
    key = np.empty(n, np.int64)
    bo = np.empty(n); bh = np.empty(n); bl = np.empty(n); bc = np.empty(n)
    ao = np.empty(n); ah = np.empty(n); al = np.empty(n); ac = np.empty(n)
    cnt = np.empty(n, np.int64)
    m = -1
    cur = -1
    for i in range(n):
        k = ts[i] // 60000
        b = bid[i] / scale
        a = ask[i] / scale
        if k != cur:
            m += 1; cur = k
            key[m] = k
            bo[m] = b; bh[m] = b; bl[m] = b; bc[m] = b
            ao[m] = a; ah[m] = a; al[m] = a; ac[m] = a
            cnt[m] = 0
        else:
            if b > bh[m]: bh[m] = b
            if b < bl[m]: bl[m] = b
            bc[m] = b
            if a > ah[m]: ah[m] = a
            if a < al[m]: al[m] = a
            ac[m] = a
        cnt[m] += 1
    m += 1
    return key[:m], bo[:m], bh[:m], bl[:m], bc[:m], ao[:m], ah[:m], al[:m], ac[:m], cnt[:m]

def bars_from_month_files(files):
    out = [[] for _ in range(10)]
    for f in files:
        z = np.load(f)
        res = m1_bars_bidask(z["ts"], z["bid"], z["ask"], PRICE_SCALE)
        for j, arr in enumerate(res):
            out[j].append(arr)
    return [np.concatenate(x) for x in out]

def save_bars(path, cols):
    names = ["key", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "cnt"]
    np.savez(path, **dict(zip(names, cols)))

def load_bars(path):
    z = np.load(path)
    return [z[k] for k in ["key", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "cnt"]]

def weekday_tow(key):
    """key = minute index since epoch -> (weekday Mon=0..Sun=6, ms of week)"""
    ms = key * MS_MIN
    d = ms // MS_DAY
    wd = (d + 3) % 7
    tow = wd * MS_DAY + (ms % MS_DAY)
    return wd, tow
