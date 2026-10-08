"""Stream EURUSD FX-Data yearly branches and aggregate the bid/ask VOLUME fields per M5 bucket
(bucket = epoch_ms // 300000). Each hourly file holds ticks sorted by time, and an M5 bucket never spans
two hourly files, so per-file reduceat aggregation is exact.
Outputs per bucket: n, sbv, sav (sums), fbv, fav (first tick), lbv, lav (last tick)."""
import sys, subprocess, tarfile, time, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev")
from parse_test import parse_tick_csv
REPO = "FX-Data/FX-Data-EURUSD-DS"
SCALE = 100000.0
years = [int(y) for y in sys.argv[1].split(",")]
out = sys.argv[2]
rows = []
for yr in years:
    t0 = time.time()
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/EURUSD-{yr}"
    proc = subprocess.Popen(["curl", "-sSfL", "--retry", "3", "--retry-delay", "5", "-m", "3000", url], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
    nf = 0; nt = 0
    for m in tf:
        if not (m.isfile() and m.name.endswith("_ticks.csv")):
            continue
        data = tf.extractfile(m).read()
        nf += 1
        if not data:
            continue
        buf = np.frombuffer(data, dtype=np.uint8)
        ts, bid, ask, bv, av = parse_tick_csv(buf, SCALE)
        if len(ts) == 0:
            continue
        o = np.argsort(ts, kind="stable")
        ts = ts[o]; bv = bv[o].astype(np.float64); av = av[o].astype(np.float64)
        b = ts // 300000
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        ends = np.r_[starts[1:] - 1, len(b) - 1]
        keys = b[starts]
        n = (ends - starts + 1).astype(np.float64)
        sbv = np.add.reduceat(bv, starts); sav = np.add.reduceat(av, starts)
        rows.append(pd.DataFrame({"bucket": keys, "n": n, "sbv": sbv, "sav": sav,
                                  "fbv": bv[starts], "fav": av[starts], "lbv": bv[ends], "lav": av[ends]}))
        nt += len(ts)
    tf.close(); proc.wait()
    print(f"year {yr}: files {nf}, ticks {nt:,}, {time.time()-t0:.0f}s", flush=True)
df = pd.concat(rows, ignore_index=True).sort_values("bucket")
df.to_parquet(out, index=False)
print("written", out, len(df), "buckets", flush=True)
