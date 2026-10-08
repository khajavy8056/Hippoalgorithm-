"""Build a memmap tick store (ts int64 ms, bid/ask int32 points) from monthly npz files,
plus an M1 bar frame cache (mid OHLC, tick counts, spread stats, up/down tick counts).
usage: build_store.py <months_dir> <full_dir> <m1_parquet_out>   (full_dir may already exist -> skipped)"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev")
from core import build_m1, MS_MIN, PTS
from signals import build_m1_frame

months_dir, full_dir, m1_out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(full_dir, exist_ok=True)
t0 = time.time()
if not os.path.exists(os.path.join(full_dir, "meta.json")):
    files = sorted(glob.glob(os.path.join(months_dir, "*.npz")))
    lens = []
    for f in files:
        with np.load(f) as z:
            lens.append(len(z["ts"]))
    total = int(sum(lens))
    ts_m = np.lib.format.open_memmap(os.path.join(full_dir, "ts.npy"), mode="w+", dtype=np.int64, shape=(total,))
    bid_m = np.lib.format.open_memmap(os.path.join(full_dir, "bid.npy"), mode="w+", dtype=np.int32, shape=(total,))
    ask_m = np.lib.format.open_memmap(os.path.join(full_dir, "ask.npy"), mode="w+", dtype=np.int32, shape=(total,))
    off, prev_last = 0, None
    for f in files:
        with np.load(f) as z:
            ts, bid, ask = z["ts"], z["bid"], z["ask"]
        if prev_last is not None and len(ts):
            assert ts[0] >= prev_last, f"out of order {f}"
        n = len(ts)
        ts_m[off:off + n] = ts; bid_m[off:off + n] = bid; ask_m[off:off + n] = ask
        off += n
        if n:
            prev_last = ts[-1]
    ts_m.flush(); bid_m.flush(); ask_m.flush(); del ts_m, bid_m, ask_m
    meta = {"n_ticks": total, "months": [os.path.basename(f)[:6] for f in files]}
    with open(os.path.join(full_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("full store written", total, "ticks in", round(time.time() - t0, 1), "s", flush=True)
ts = np.load(os.path.join(full_dir, "ts.npy"), mmap_mode="r")
bid = np.load(os.path.join(full_dir, "bid.npy"), mmap_mode="r")
ask = np.load(os.path.join(full_dir, "ask.npy"), mmap_mode="r")
if not os.path.exists(m1_out):
    # build M1 in monthly chunks to bound memory
    frames = []
    t_first = pd.Timestamp(int(ts[0]) // 1000, unit="s").to_period("M").to_timestamp()
    t_last = pd.Timestamp(int(ts[-1]) // 1000, unit="s")
    cur = t_first
    while cur <= t_last:
        nxt = cur + pd.offsets.MonthBegin(1)
        lo = np.searchsorted(ts, int(cur.value // 10**6))
        hi = np.searchsorted(ts, int(nxt.value // 10**6))
        if hi > lo:
            m1 = build_m1_frame(np.asarray(ts[lo:hi]), np.asarray(bid[lo:hi]), np.asarray(ask[lo:hi]))
            frames.append(m1)
        cur = nxt
    print("  M1 chunks", len(frames), round(time.time() - t0, 1), "s", flush=True)
    M1 = pd.concat(frames)
    M1 = M1[~M1.index.duplicated()]
    M1.to_parquet(m1_out)
    print("M1 written", len(M1), "bars ->", m1_out, round(time.time() - t0, 1), "s", flush=True)
