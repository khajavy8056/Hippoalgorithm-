"""Family V: same M5 direct classifier as fam_ml, plus QUOTE-VOLUME (top-of-book depth) features aggregated per M5 bar.
Volume features use only the bar's own ticks (available at the bar close) and lagged bars; no look-ahead."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2")
import fam_ml as M

VOL = pd.concat([pd.read_parquet("/tmp/hipo_r2/vol_A.parquet"), pd.read_parquet("/tmp/hipo_r2/vol_B.parquet")], ignore_index=True)
VOL = VOL.drop_duplicates("bucket").set_index("bucket").sort_index()
VOL_COLS = ["vimb", "vimb_last", "vimb_chg3", "vimb_lag1", "vimb_lag2", "vtot_rel", "vbid_rel", "vask_rel",
            "vdepth_chg_bid", "vdepth_chg_ask"]


def features_vol(name):
    F, feat = M.features_base(name)
    key = (F["dec_ms"].values.astype(np.int64) - 300000) // 300000
    v = VOL.reindex(key)
    sb, sa = v["sbv"].values, v["sav"].values
    tot = sb + sa
    G = pd.DataFrame(index=F.index)
    G["vimb"] = (sa - sb) / tot
    G["vimb_last"] = (v["lav"].values - v["lbv"].values) / (v["lav"].values + v["lbv"].values)
    G["vimb_chg3"] = G["vimb"] - G["vimb"].shift(3)
    G["vimb_lag1"] = G["vimb"].shift(1)
    G["vimb_lag2"] = G["vimb"].shift(2)
    med_tot = pd.Series(tot, index=F.index).rolling(288, min_periods=50).median()
    G["vtot_rel"] = tot / med_tot.values
    G["vbid_rel"] = sb / pd.Series(sb, index=F.index).rolling(288, min_periods=50).median().values
    G["vask_rel"] = sa / pd.Series(sa, index=F.index).rolling(288, min_periods=50).median().values
    G["vdepth_chg_bid"] = (v["lbv"].values - v["fbv"].values) / (v["lbv"].values + v["fbv"].values)
    G["vdepth_chg_ask"] = (v["lav"].values - v["fav"].values) / (v["lav"].values + v["fav"].values)
    F2 = pd.concat([F, G[VOL_COLS]], axis=1)
    return F2, feat + VOL_COLS


M.features = features_vol   # block() in fam_ml resolves features() at call time

if __name__ == "__main__":
    stage = sys.argv[1]
    cfg = eval(sys.argv[2])
    tag = "vol_" + "_".join(str(x) for x in cfg)
    if stage == "disc":
        T, aucs = M.wf_disc(cfg)
        s_ = M.summarize_trades(T, "VOL-WF " + str(cfg))
        from frontier import approx_frontier
        fr = approx_frontier(T) if len(T) >= 5 else {}
        print("DISC-WF", s_, {k: round(float(v), 4) for k, v in fr.items() if v == v}, "AUC(test,dir):", aucs, flush=True)
        if len(T):
            T.to_parquet(f"/tmp/hipo_r2/results/{tag}_disc.parquet")
    else:
        T, aucs, thrs = M.conf_run(cfg)
        s_ = M.summarize_trades(T, "VOL-CONF " + str(cfg))
        print("CONF", s_, "AUC:", aucs, "thr:", thrs, flush=True)
        if len(T):
            T.to_parquet(f"/tmp/hipo_r2/results/{tag}_conf.parquet")
