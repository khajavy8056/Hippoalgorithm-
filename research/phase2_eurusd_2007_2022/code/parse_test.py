import numpy as np, pandas as pd, time
from numba import njit


@njit(cache=True)
def _days_from_civil(y, m, d):
    # Howard Hinnant's algorithm: days since 1970-01-01
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


@njit(cache=True)
def parse_tick_csv(buf, scale):
    """Parse 'YYYY.MM.DD HH:MM:SS.mmm,bid,ask,bv,av' lines.
    Returns ts_ms (int64, UTC epoch ms), bid_pts, ask_pts (int64 price*scale rounded), bv, av (float32)."""
    n = buf.shape[0]
    cap = 1
    for k in range(n):
        if buf[k] == 10:
            cap += 1
    ts = np.empty(cap, np.int64)
    bid = np.empty(cap, np.int64)
    ask = np.empty(cap, np.int64)
    bv = np.empty(cap, np.float32)
    av = np.empty(cap, np.float32)
    cnt = 0
    i = 0
    while i < n:
        # ---- time field: collect digits -> YYYYMMDDHHMMSSmmm (17 digits)
        digs = np.int64(0)
        nd = 0
        bad = False
        while i < n and buf[i] != 44:  # ','
            c = buf[i]
            if buf[i] == 10:
                bad = True
                break
            if 48 <= c <= 57:
                if nd < 17:
                    digs = digs * 10 + (c - 48)
                nd += 1
            i += 1
        if bad or i >= n:
            # malformed line (or header) -> skip it
            while i < n and buf[i] != 10:
                i += 1
            i += 1
            continue
        i += 1  # skip ','
        if nd != 17:
            while i < n and buf[i] != 10:
                i += 1
            i += 1
            continue
        ms = digs % 1000
        t = digs // 1000
        ss = t % 100
        t //= 100
        mi = t % 100
        t //= 100
        hh = t % 100
        t //= 100
        dd = t % 100
        t //= 100
        mo = t % 100
        t //= 100
        yy = t
        days = _days_from_civil(yy, mo, dd)
        ts_val = (days * 86400 + hh * 3600 + mi * 60 + ss) * 1000 + ms
        # ---- 4 numeric fields: bid, ask, bv, av
        vals = np.zeros(4, np.float64)
        ok = True
        for f in range(4):
            ip = np.int64(0)
            fp = np.int64(0)
            fd = 0
            seen_dot = False
            nd2 = 0
            while i < n and buf[i] != 44 and buf[i] != 10 and buf[i] != 13:
                c = buf[i]
                if c == 46:
                    seen_dot = True
                elif 48 <= c <= 57:
                    if seen_dot:
                        if fd < 9:
                            fp = fp * 10 + (c - 48)
                            fd += 1
                    else:
                        ip = ip * 10 + (c - 48)
                    nd2 += 1
                else:
                    ok = False
                i += 1
            if nd2 == 0 and f < 2:
                ok = False  # bid/ask required; volumes optional (HistData has 3 fields)
            v = np.float64(ip)
            if fd > 0:
                p10 = 1.0
                for _ in range(fd):
                    p10 *= 10.0
                v += np.float64(fp) / p10
            vals[f] = v
            if i < n and buf[i] == 44:
                i += 1
        while i < n and buf[i] != 10:
            i += 1
        i += 1
        if not ok:
            continue
        ts[cnt] = ts_val
        bid[cnt] = np.int64(np.floor(vals[0] * scale + 0.5))
        ask[cnt] = np.int64(np.floor(vals[1] * scale + 0.5))
        bv[cnt] = np.float32(vals[2])
        av[cnt] = np.float32(vals[3])
        cnt += 1
    return ts[:cnt], bid[:cnt], ask[:cnt], bv[:cnt], av[:cnt]


if __name__ == "__main__":
    fn = "/tmp/hipo_raw/sample_2019-01-02_08h.csv"
    raw = open(fn, "rb").read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    r = parse_tick_csv(buf, 100000.0)
    t1 = time.time()
    r = parse_tick_csv(buf, 100000.0)
    t2 = time.time()
    print("numba parse rows", len(r[0]), "time(2nd call)", round(t2 - t1, 4))
    df = pd.read_csv(fn, header=None, names=["t", "bid", "ask", "bv", "av"])
    tt = pd.to_datetime(df["t"], format="%Y.%m.%d %H:%M:%S.%f")
    ref_ts = (tt.astype("int64") // 10**6).to_numpy()
    print("ts equal:", np.array_equal(ref_ts, r[0]))
    print("bid pts equal:", np.array_equal(np.round(df["bid"].values * 1e5).astype(np.int64), r[1]))
    print("ask pts equal:", np.array_equal(np.round(df["ask"].values * 1e5).astype(np.int64), r[2]))
    print("bv max abs diff:", float(np.max(np.abs(df["bv"].values - r[3]))), " av:", float(np.max(np.abs(df["av"].values - r[4]))))
    print(r[0][:3], r[1][:3], r[2][:3], r[3][:3], r[4][:3])
    print("first ts ->", pd.to_datetime(r[0][0], unit="ms"), " last ->", pd.to_datetime(r[0][-1], unit="ms"))
