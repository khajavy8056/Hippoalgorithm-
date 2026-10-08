"""Discovery-period ETL (2007-2016): stream FX-Data EURUSD yearly branches, parse ticks,
build M1 bid/ask OHLC bars per year and save them. Tick arrays are NOT kept (disk budget)."""
import io, json, os, subprocess, sys, tarfile, time, datetime as dt
import numpy as np
sys.path.insert(0, "/tmp/hipo_dev"); sys.path.insert(0, "/tmp/hipo_r2")
from parse_test import parse_tick_csv
from m1bars import save_bars

SCALE = 100000.0
REPO = "FX-Data/FX-Data-EURUSD-DS"
YEARS = [int(y) for y in os.environ.get("ETL_YEARS", "2007,2008,2009,2010,2011,2012,2013,2014,2015,2016").split(",")]
OUTDIR = "/tmp/hipo_r2/bars"
os.makedirs(OUTDIR, exist_ok=True)

def log(msg):
    print(f"[{dt.datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)

def m1_from_arrays(ts, bid, ask):
    from m1bars import m1_bars_bidask
    return m1_bars_bidask(ts, bid, ask, SCALE)

for year in YEARS:
    out_path = f"{OUTDIR}/m1_{year}.npz"
    if os.path.exists(out_path):
        log(f"{year}: exists, skip"); continue
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/EURUSD-{year}"
    log(f"{year}: streaming {url}")
    t0 = time.time()
    proc = subprocess.Popen(["curl", "-sSfL", "--retry", "3", "--retry-delay", "5", "-m", "3000", url],
                            stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
    TS, BID, ASK = [], [], []
    qa = dict(files=0, raw_rows=0, crossed_dropped=0, malformed_or_header=0)
    for m in tf:
        if not (m.isfile() and m.name.endswith("_ticks.csv")):
            continue
        data = tf.extractfile(m).read()
        qa["files"] += 1
        if not data:
            continue
        buf = np.frombuffer(data, dtype=np.uint8)
        ts, bid, ask, bv, av = parse_tick_csv(buf, SCALE)
        qa["raw_rows"] += int(len(ts))
        if len(ts) == 0:
            continue
        crossed = (bid > ask) | (bid <= 0) | (ask <= 0)
        qa["crossed_dropped"] += int(crossed.sum())
        keep = ~crossed
        TS.append(ts[keep]); BID.append(bid[keep].astype(np.int32)); ASK.append(ask[keep].astype(np.int32))
    proc.stdout.close(); proc.kill()
    ts = np.concatenate(TS); bid = np.concatenate(BID); ask = np.concatenate(ASK)
    TS = BID = ASK = None
    order = np.argsort(ts, kind="stable")
    ts = ts[order]; bid = bid[order]; ask = ask[order]
    dup = int((np.diff(ts) == 0).sum())
    gaps = np.diff(ts)
    qa.update(ticks=int(len(ts)), dup_ms=dup, gaps_gt_30min=int((gaps > 30*60000).sum()),
              first=str(dt.datetime.utcfromtimestamp(ts[0]/1000)), last=str(dt.datetime.utcfromtimestamp(ts[-1]/1000)),
              secs=round(time.time()-t0, 1))
    cols = m1_from_arrays(ts, bid, ask)
    save_bars(out_path, cols)
    sp = (cols[5] - cols[4]) * 1e4
    qa.update(bars=int(len(cols[0])), median_spread_pips=round(float(np.median(sp)), 3),
              p99_spread_pips=round(float(np.percentile(sp, 99)), 3))
    log(f"{year}: " + json.dumps(qa))
    with open(f"{OUTDIR}/qa_{year}.json", "w") as f:
        json.dump(qa, f, indent=1)
    ts = bid = ask = order = None
log("ALL DONE")
