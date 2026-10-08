import sys, json
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np
from hipo_core import scan_direction, params_vector, OUT_COLS

P = params_vector()

def arrays(bars):
    a = np.array(bars, float)
    return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy()

def run(bars, atr, direction=+1, **kw):
    O, H, L, C = arrays(bars)
    ATR = np.full(len(bars), float(atr))
    res = scan_direction(O, H, L, C, ATR, params_vector(**kw), direction)
    return [dict(zip(OUT_COLS, r)) for r in res]

BASE = [
    (100, 108, 100, 107),   # 0  A-bar (bull)
    (107, 120, 106, 119),   # 1  AB (bull)
    (119, 132, 118, 131),   # 2  AB (bull)
    (131, 145, 130, 144),   # 3  B-bar: high 145 (bull)
    (144, 144.5, 134, 135), # 4  BC (bear)  body retrace 22%
    (135, 137, 131.5, 132.5),  # 5 BC (bear)
    (132.5, 133.5, 130, 131.5),# 6 BC (bear) C = 130 (lowest low)
    (131.5, 140, 131, 139.5),  # 7 CD (bull) reversal
    (142, 149, 140, 148),   # 8  signal candle: breaks B(145), straddles B, small upper wick
    (148, 150, 146, 149.5), # 9  candle closes above signal high -> minor high 150
    (147.5, 148, 136, 137), # 10 confirmation: body closes below signal low 140
]
results = {}
def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))

# ---- positive base case -------------------------------------------------------------
r = run(BASE, 10.0)
check("base: exactly one short signal", len(r) == 1, f"n={len(r)}")
if r:
    s = r[0]
    check("base: entry bar = confirmation bar", s["entry_idx"] == 10)
    check("base: structure A,B,C,S = 0,3,6,8", (s["a_idx"], s["b_idx"], s["c_idx"], s["sig_idx"]) == (0, 3, 6, 8))
    check("base: entry = close of confirmation (137)", abs(s["entry"] - 137.0) < 1e-9)
    check("base: SL above signal high incl. minor high 150 + 0.25 ATR = 152.5", abs(s["sl"] - 152.5) < 1e-9, f"sl={s['sl']}")
    check("base: TP = entry - 2.5R = 98.25", abs(s["tp"] - 98.25) < 1e-9, f"tp={s['tp']:.3f}")
    check("base: AB has 4 candles (>3 -> 200-500% ATR rule), size 45", s["ab_bars"] == 4 and abs(s["ab_size"] - 45) < 1e-9)

# ---- mirror (long setup) -----------------------------------------------------------
M = [(300 - o, 300 - l, 300 - h, 300 - c) for (o, h, l, c) in BASE]    # reflect around 150
r = run(M, 10.0, direction=-1)
check("mirror: one long signal with mirrored levels", len(r) == 1 and abs(r[0]["entry"] - 163.0) < 1e-9
      and abs(r[0]["sl"] - 147.5) < 1e-9 and abs(r[0]["tp"] - 201.75) < 1e-9,
      f"{[(x['entry'], x['sl'], x['tp']) for x in r]}")

# ---- negative tests (each must remove the trade at bar 10) -------------------------
def mutate(idx, new):
    b = list(BASE); b[idx] = new; return b

cases = {
  "AB: two candles, each < ATR -> reject":
      [(100, 108, 100, 107), (107, 145, 107, 144)] + BASE[4:],          # bar0 range 8 < ATR 10
  "AB: >3 candles but size > 5 ATR (60 > 50) -> reject":
      mutate(3, (131, 160, 130, 144)),
  "BC: body closes beyond 50% of AB -> reject":
      mutate(6, (132.5, 133.5, 130, 121)),
  "BC: wick touches A (99.5 <= 100) -> reject":
      mutate(6, (132.5, 133.5, 99.5, 131.5)),
  "CD >= AB (box top 175 broken) -> reject":
      mutate(7, (131.5, 180, 131, 179)),
  "signal: large upper wick -> reject":
      mutate(8, (141, 149, 140, 141.5)),
  "signal: range < 0.8 ATR -> reject (bar 9 then cannot be signal)":
      mutate(8, (142, 146, 141, 145)),
  "confirmation body < 0.5 ATR -> no entry":
      mutate(10, (137.5, 148, 136, 136.5)),
  "close beyond liquidity box before entry -> setup fails":
      mutate(9, (148, 178, 146, 176)),
}
for name, bars in cases.items():
    r = run(bars, 10.0)
    ok = not any(x["entry_idx"] == 10 for x in r)
    check(name, ok, f"signals={[(x['entry_idx'], round(x['entry'],2)) for x in r]}")

# ---- golden: NAS100 example from the PDF (bars digitised from the chart, ATR14=14.02) --
NAS = [
 (30702.37, 30718.56, 30697.16, 30714.09),  # b0  20:51 A-candle (low = A)
 (30714.47, 30720.98, 30711.86, 30720.98),  # b1
 (30721.72, 30734.93, 30721.72, 30733.07),  # b2
 (30733.07, 30755.40, 30730.28, 30746.65),  # b3  B-candle (high = B = 30755.4)
 (30746.28, 30748.14, 30737.91, 30740.51),  # b4  BC
 (30739.95, 30742.37, 30732.70, 30736.42),  # b5  BC
 (30735.67, 30739.77, 30718.56, 30733.07),  # b6  C (low 30718.56)
 (30733.81, 30750.00, 30728.60, 30745.53),  # b7  CD
 (30745.91, 30748.51, 30743.30, 30748.51),  # b8  small candle below B
 (30748.88, 30763.95, 30746.65, 30760.80),  # b9  SIGNAL candle (high breaks B, low 30746.65)
 (30762.09, 30769.72, 30756.88, 30766.56),  # b10 D-candle
 (30766.56, 30768.42, 30758.00, 30760.98),  # b11
 (30760.23, 30762.65, 30752.98, 30756.51),  # b12
 (30755.95, 30760.05, 30734.93, 30742.93),  # b13 CONFIRMATION (closes below 30746.65)
]
r = run(NAS, 14.02)
check("NAS golden: one short signal at confirmation candle b13", len(r) == 1 and r[0]["entry_idx"] == 13,
      f"{[(x['entry_idx'], x['sig_idx']) for x in r]}")
if r:
    s = r[0]
    print("   NAS golden detail:", {k: round(s[k], 2) for k in ["entry", "sl", "tp", "ab_size", "minor_hi"]},
          "R=", round(s["sl"] - s["entry"], 2), "| trader drew SL 30774.55, entry 30743.13, TP 30676.88 (2.1R)")
    check("NAS golden: entry within 0.5 pt of chart entry (30743.13)", abs(s["entry"] - 30743.13) < 0.5)
    check("NAS golden: SL within 2 pt of drawn SL (30774.55)", abs(s["sl"] - 30774.55) < 2.0, f"sl={s['sl']:.2f}")
    check("NAS golden: TP at 2.5R (PDF) vs drawn 2.1R", abs((s["entry"] - s["tp"]) / (s["sl"] - s["entry"]) - 2.5) < 1e-6)

print("\nSUMMARY:", sum(results.values()), "/", len(results), "checks passed")
json.dump(results, open("/tmp/hipo_r2/test_results.json", "w"), indent=1)
