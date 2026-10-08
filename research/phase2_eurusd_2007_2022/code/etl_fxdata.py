"""Stream Dukascopy-derived EURUSD hourly tick CSVs from the FX-Data/FX-Data-EURUSD-DS
GitHub mirror (one branch per year), parse them with a numba parser and write
per-month compact arrays (ts_ms int64, bid/ask int32 in 1e-5 price points).
"""
import io, json, os, subprocess, sys, tarfile, time, datetime as dt
import numpy as np

sys.path.insert(0, "/tmp/hipo_dev")
from parse_test import parse_tick_csv  # verified against pandas

SCALE = 100000.0
OUT = os.environ.get("ETL_OUT", "/tmp/hipo_data")
PARTS = os.path.join(OUT, "parts")
MONTHS = os.path.join(OUT, "months")
YEARS = [int(y) for y in os.environ.get("ETL_YEARS", "2017,2018,2019,2020,2021,2022").split(",")]
REPO = "FX-Data/FX-Data-EURUSD-DS"


def log(msg):
    print(f"[{dt.datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def month_keys(ts):
    d = (ts // 1000).astype("datetime64[s]").astype("datetime64[M]")
    return d.astype(np.int64)  # months since 1970-01


def stream_year(year, qa):
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/EURUSD-{year}"
    log(f"year {year}: streaming {url}")
    t0 = time.time()
    # curl (system CA store) piped into a streaming tar reader
    proc = subprocess.Popen(["curl", "-sSfL", "--retry", "3", "--retry-delay", "5", "-m", "3000", url],
                            stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
    n_files = 0
    buckets = {}  # month_key -> list of (ts,bid,ask)
    for m in tf:
        if not (m.isfile() and m.name.endswith("_ticks.csv")):
            continue
        data = tf.extractfile(m).read()
        n_files += 1
        if not data:
            continue
        buf = np.frombuffer(data, dtype=np.uint8)
        ts, bid, ask, bv, av = parse_tick_csv(buf, SCALE)
        qa["raw_rows"] += int(len(ts))
        if len(ts) == 0:
            continue
        # crossed / zero-or-negative quotes
        crossed = (bid > ask) | (bid <= 0) | (ask <= 0)
        qa["crossed_dropped"] += int(crossed.sum())
        keep = ~crossed
        ts, bid, ask = ts[keep], bid[keep], ask[keep]
        if len(ts) == 0:
            continue
        mk = month_keys(ts)
        for k in np.unique(mk):
            sel = mk == k
            buckets.setdefault(int(k), []).append((ts[sel], bid[sel].astype(np.int32), ask[sel].astype(np.int32)))
        if n_files % 500 == 0:
            log(f"  year {year}: {n_files} hourly files, {qa['raw_rows']:,} raw rows, {time.time()-t0:.0f}s")
    tf.close()
    rc = proc.wait()
    if rc != 0 or n_files == 0:
        raise RuntimeError(f"curl/tar failed for {year}: rc={rc}, files={n_files}")
    log(f"year {year}: done {n_files} files in {time.time()-t0:.0f}s")
    os.makedirs(PARTS, exist_ok=True)
    for k, chunks in buckets.items():
        ts = np.concatenate([c[0] for c in chunks])
        bid = np.concatenate([c[1] for c in chunks])
        ask = np.concatenate([c[2] for c in chunks])
        y = 1970 + k // 12
        mo = k % 12 + 1
        fn = os.path.join(PARTS, f"{y}{mo:02d}_y{year}.npz")
        np.savez(fn, ts=ts, bid=bid, ask=ask)
        qa["months_written"].append(f"{y}{mo:02d}")
    return n_files


def consolidate():
    os.makedirs(MONTHS, exist_ok=True)
    files = sorted(os.listdir(PARTS))
    by_month = {}
    for f in files:
        by_month.setdefault(f[:6], []).append(os.path.join(PARTS, f))
    report = {}
    for ym in sorted(by_month):
        ts_l, b_l, a_l = [], [], []
        for fn in by_month[ym]:
            z = np.load(fn)
            ts_l.append(z["ts"]); b_l.append(z["bid"]); a_l.append(z["ask"])
        ts = np.concatenate(ts_l); bid = np.concatenate(b_l); ask = np.concatenate(a_l)
        order = np.argsort(ts, kind="stable")
        ts, bid, ask = ts[order], bid[order], ask[order]
        # drop exact duplicate rows (same ms, same quotes)
        dup = np.zeros(len(ts), dtype=bool)
        dup[1:] = (ts[1:] == ts[:-1]) & (bid[1:] == bid[:-1]) & (ask[1:] == ask[:-1])
        n_dup = int(dup.sum())
        ts, bid, ask = ts[~dup], bid[~dup], ask[~dup]
        np.savez(os.path.join(MONTHS, f"{ym}.npz"), ts=ts, bid=bid, ask=ask)
        report[ym] = {"rows": int(len(ts)), "duplicates_dropped": n_dup}
        log(f"month {ym}: rows={len(ts):,} dups_dropped={n_dup:,}")
    with open(os.path.join(OUT, "etl_month_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    qa = {"raw_rows": 0, "crossed_dropped": 0, "months_written": []}
    for y in YEARS:
        stream_year(y, qa)
    log(f"raw rows total={qa['raw_rows']:,} crossed/zero dropped={qa['crossed_dropped']:,}")
    consolidate()
    with open(os.path.join(OUT, "etl_qa.json"), "w") as fh:
        json.dump({k: v for k, v in qa.items() if k != 'months_written'}, fh, indent=1)
    log("ETL finished")
