import sys
sys.path.insert(0, "/tmp/hipo_r2")
import numpy as np, pandas as pd
from common_r2 import *
from disc_grid import load_years, max_stop_atr_filter
from disc_ofat import filt, session_mask
from gross_diag import gross_outcomes
m1 = load_years(list(range(2007, 2017)))
kw = dict(TP_R=2.5, SL_BUF_ATR=0.25, CONF_BODY_ATR=0.8)
sig = max_stop_atr_filter(signals_for_tf(m1, 5, kw), 2.0)
sig = filt(sig, session_mask(m1, sig))
n = len(sig["k"])
mh = (m1[2] + m1[6]) / 2; ml = (m1[3] + m1[7]) / 2
o = gross_outcomes(mh, ml, sig["k"], sig["dir"], sig["sl"], sig["tp"], 1440)
gR = float(np.nanmean(o[:, 0]))
rows = []
for risk in (0.0025, 0.005, 0.01, 0.015, 0.02):
    tr = run_bar_backtest(m1, sig, risk)
    met = metrics_from_trades(tr)
    rows.append(dict(variant="V2+V4 combo (CONF_BODY 0.8 + 07-21 UTC)", risk_pct=risk*100, signals=n, gross_mean_R=gR, **met))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df[["variant","risk_pct","signals","gross_mean_R","trades","win_pct","mean_R","pf","total_pct","monthly_geo_pct","max_dd_pct","worst_month_pct","pos_months_pct"]].round(3).to_string(index=False))
df.to_csv("/tmp/hipo_r2/disc_combo_results.csv", index=False)
o2 = pd.read_csv("/tmp/hipo_r2/disc_ofat_results.csv")
sel = o2[o2.variant.str.startswith(("V4","V2","V9","base"))][["variant","risk_pct","signals","gross_mean_R","trades","mean_R","monthly_geo_pct","max_dd_pct"]]
print(sel.round(3).to_string(index=False))
