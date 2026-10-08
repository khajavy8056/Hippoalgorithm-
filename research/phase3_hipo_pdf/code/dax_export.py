"""Export digitised candles (pixel-price units, price = -y) and drawn trade levels from the trade-example images."""
import sys, glob, os
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from dax_v3 import digitise_v3, drawn_from_zones
from dax_digitize import load_rgb
OUT = "/home/user/Hippoalgorithm-/research/phase3_hipo_pdf/data"
files = sorted(f for f in glob.glob("/tmp/hipo_doc/DAX-*.png") if "(1)" not in f)
cand_rows, lvl_rows = [], []
for f in files:
    name = os.path.basename(f)
    im = load_rgb(f)
    dg = digitise_v3(im)
    for k, c in enumerate(dg["candles"]):
        cand_rows.append(dict(image=name, k=k, x=c["xc"], bullish=int(c["bullish"]), body_top=c["top"], body_bot=c["bot"],
                              wick_top=c["wtop"], wick_bot=c["wbot"]))
    dt = drawn_from_zones(im)
    row = dict(image=name, spacing_px=round(dg["d"], 3), n_candles=len(dg["candles"]))
    if dt is not None:
        row.update(drawn_dir=(None if dt.get("dir") is None else ("SHORT" if dt["dir"] < 0 else "LONG")),
                   drawn_x=dt.get("xl"), entry_y=dt.get("entry_y"), sl_y=dt.get("sl_y"), tp_y=dt.get("tp_y"),
                   drawn_R_px=dt.get("R_px"))
    lvl_rows.append(row)
pd.DataFrame(cand_rows).to_csv(f"{OUT}/dax_candles_px.csv", index=False)
pd.DataFrame(lvl_rows).to_csv(f"{OUT}/dax_levels_px.csv", index=False)
print(pd.DataFrame(lvl_rows).to_string(index=False))
print("candles:", len(cand_rows))
