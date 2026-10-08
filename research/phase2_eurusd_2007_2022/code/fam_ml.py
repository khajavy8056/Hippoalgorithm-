"""Family M: high-frequency (M5) direct classifier of cost-aware triple-barrier outcomes, per direction.
Label: core.candidate_outcomes (real bid/ask, spread, slippage, commission): SL=s*ATR5, TP=rr*SL, time-stop H bars.
y = 1 if net pips > 0. Causal features only. NaN kept (HistGradientBoosting handles missing natively; no fillna(0)).
Model per direction. Inside each training window: first 80% fit, last 20% validation (threshold choice),
purge = H bars + 1 day before the test start.
DISC walk-forward: test years 2011..2016, training from 2007.
CONF: one frozen fit on DISC (2007..2016), test 2017-01..2022-07 (never used for any choice).
usage: fam_ml.py disc   |   fam_ml.py conf "(s,rr,H)"
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/hipo_dev/r2"); sys.path.insert(0, "/tmp/hipo_dev")
from common import load_m1, load_ticks, bars, PRM
from core import candidate_outcomes, MS_MIN, MS_HOUR, MS_DAY, P_SPREADMULT, P_SLIP, P_COMM, P_MAXSPREAD
PRM_GROSS = PRM.copy(); PRM_GROSS[P_SPREADMULT] = 0.0; PRM_GROSS[P_SLIP] = 0.0; PRM_GROSS[P_COMM] = 0.0; PRM_GROSS[P_MAXSPREAD] = 0.0  # mid-to-mid, no costs
from signals import atr, rsi, adx
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from frontier import approx_frontier

CFGS = [(1.5, 1.0, 12), (2.0, 1.5, 48)]   # (SL in ATR5, RR, horizon in M5 bars)


def ms(y, m=1, d=1):
    return int(pd.Timestamp(year=y, month=m, day=d, tz="UTC").value // 10**6)


def features_base(name):
    m1 = load_m1(name)
    b5 = bars(m1, "5min")
    h1 = bars(m1, "1h")
    c, hi, lo = b5["close"], b5["high"], b5["low"]
    A = atr(b5, 14)
    F = pd.DataFrame(index=b5.index)
    for k in (1, 2, 3, 6, 12, 24, 48):
        F[f"ret{k}"] = (c - c.shift(k)) / A
    F["atrp"] = A / c * 1e4
    F["volratio"] = A / A.rolling(288, min_periods=100).mean()
    F["rng"] = (hi - lo) / A
    F["rsi14"] = rsi(c, 14)
    F["rsi6"] = rsi(c, 6)
    F["adx14"] = adx(b5, 14)
    e20 = c.ewm(span=20, adjust=False).mean(); e100 = c.ewm(span=100, adjust=False).mean()
    F["dema20"] = (c - e20) / A
    F["dema100"] = (c - e100) / A
    F["pos48"] = (c - lo.rolling(48).min()) / (hi.rolling(48).max() - lo.rolling(48).min())
    F["imb"] = (b5["up"] - b5["dn"]) / (b5["up"] + b5["dn"] + 1)
    F["act"] = b5["ticks"] / b5["ticks"].rolling(288, min_periods=50).median()
    F["spr"] = b5["spread"]
    F["spr_rel"] = b5["spread"] / b5["spread"].rolling(288, min_periods=50).median()
    dec = b5["dec_ms"].values.astype(np.int64)
    hod = (dec % MS_DAY) / MS_HOUR
    F["hod_s"] = np.sin(2 * np.pi * hod / 24)
    F["hod_c"] = np.cos(2 * np.pi * hod / 24)
    F["dow"] = ((dec // MS_DAY) + 3) % 7
    F["dec_ms"] = dec
    F["atr5"] = A.values
    A1 = atr(h1, 14)
    e50 = h1["close"].ewm(span=50, adjust=False).mean(); e200 = h1["close"].ewm(span=200, adjust=False).mean()
    H1 = pd.DataFrame({"dec_ms": h1["dec_ms"].values.astype(np.int64),
                       "h1trend": ((e50 - e200) / A1).values,
                       "h1ret6": ((h1["close"] - h1["close"].shift(6)) / A1).values})
    F = pd.merge_asof(F.sort_values("dec_ms"), H1.sort_values("dec_ms"), on="dec_ms", direction="backward")
    feat = [x for x in F.columns if x not in ("dec_ms", "atr5")]
    return F, feat


def features(name):
    return features_base(name)


def block(name, cfg):
    """features + per-direction labels for one store; rows = decision bars with valid ATR."""
    s, rr, H = cfg
    ts, bid, ask = load_ticks(name)
    F, feat = features(name)
    A = F["atr5"].values
    ok = np.isfinite(A) & (A > 0)
    G = F.loc[ok].reset_index(drop=True)
    out = {"X": G[feat].values.astype(np.float32), "dec": G["dec_ms"].values.astype(np.int64), "lab": {}, "feat": feat,
           "sl_pips": (s * G["atr5"].values) / 1e-4}
    sl = s * G["atr5"].values
    tp = rr * sl
    hold = np.full(len(G), H * 5 * MS_MIN, dtype=np.int64)
    for d in (1, -1):
        o = candidate_outcomes(ts, bid, ask, out["dec"], np.full(len(G), float(d)), sl, tp, hold, PRM)
        og = candidate_outcomes(ts, bid, ask, out["dec"], np.full(len(G), float(d)), sl, tp, hold, PRM_GROSS)
        out["lab"][d] = {"filled": o[:, 0] > 0.5, "net": o[:, 5], "R": o[:, 6], "exit": o[:, 3],
                         "gross": og[:, 5], "gR": og[:, 6]}
    return out


def concat_blocks(bl):
    X = np.concatenate([b["X"] for b in bl]); dec = np.concatenate([b["dec"] for b in bl])
    SLP = np.concatenate([b["sl_pips"] for b in bl])
    lab = {d: {k: np.concatenate([b["lab"][d][k] for b in bl]) for k in ("filled", "net", "R", "exit", "gross", "gR")} for d in (1, -1)}
    lab["sl_pips"] = SLP
    return X, dec, lab


def greedy_idx(dec, exit_ts, order_idx):
    """one-position-at-a-time subset of candidate indices (ordered by decision time)."""
    keep = []
    last = -1
    for i in order_idx:
        if dec[i] >= last:
            keep.append(i)
            last = exit_ts[i]
    return np.array(keep, dtype=np.int64)


def fit_and_thresholds(X, dec, lab, d, train_hi, H_ms, stride=2):
    """train on decisions < train_hi - H - 1d (purged); 80/20 time split; threshold on last 20%."""
    purge_end = train_hi - H_ms - MS_DAY
    t0 = dec[0]
    split = int(t0 + 0.8 * (purge_end - t0))
    L = lab[d]
    fit_m = (dec < split - H_ms) & L["filled"]
    val_m = (dec >= split) & (dec < purge_end) & L["filled"]
    fit_idx = np.where(fit_m)[0][::stride]
    y_fit = (L["net"][fit_idx] > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=300, l2_regularization=1.0, random_state=0)
    clf.fit(X[fit_idx], y_fit)
    val_idx = np.where(val_m)[0]
    p_val = clf.predict_proba(X[val_idx])[:, 1]
    best = (None, -np.inf, 0)
    for q in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
        thr = float(np.quantile(p_val, q))
        sel = val_idx[p_val >= thr]
        g = greedy_idx(dec, L["exit"], sel)
        if len(g) >= 150 and L["R"][g].mean() > best[1]:
            best = (thr, float(L["R"][g].mean()), len(g))
    return clf, best


def predict_trades(X, dec, lab, d, clf, thr, lo, hi):
    L = lab[d]
    te_idx = np.where((dec >= lo) & (dec < hi) & L["filled"])[0]
    p = clf.predict_proba(X[te_idx])[:, 1]
    sel = te_idx[p >= thr]
    yte = (L["net"][te_idx] > 0).astype(int)
    auc = roc_auc_score(yte, p) if 0 < yte.mean() < 1 else np.nan
    return sel, auc


def trades_df(dec, lab, idx, d):
    return pd.DataFrame({"dec_ms": dec[idx], "exit_ts": lab[d]["exit"][idx], "R": lab[d]["R"][idx],
                         "net_pips": lab[d]["net"][idx], "gross_R": lab[d]["gR"][idx], "gross_pips": lab[d]["gross"][idx],
                         "sl_pips": lab["sl_pips"][idx],
                         "dir": d, "filled": True})


def summarize_trades(T, label):
    if len(T) < 5:
        return {"label": label, "n": len(T)}
    R = T["R"].values
    yrs = pd.to_datetime(T["dec_ms"].values // 1000, unit="s").year
    by = pd.Series(R).groupby(np.asarray(yrs)).agg(["mean", "size"])
    return {"label": label, "n": len(T), "mean_R": R.mean(), "t": R.mean() / (R.std(ddof=1) / np.sqrt(len(R))),
            "mean_pips": T["net_pips"].mean(), "hit": (T["net_pips"] > 0).mean(),
            "GROSS_mean_pips": T["gross_pips"].mean(), "GROSS_mean_R": T["gross_R"].mean(),
            "yrs_pos": f"{int((by['mean'] > 0).sum())}/{len(by)}",
            "trades_per_yr": round(len(T) / max(1, len(by)), 1)}


def wf_disc(cfg):
    s, rr, H = cfg
    H_ms = H * 5 * MS_MIN
    B = block("disc", cfg)
    X, dec, lab = B["X"], B["dec"], B["lab"]
    lab["sl_pips"] = B["sl_pips"]
    allT = []
    aucs = []
    for y in range(2011, 2017):
        lo, hi = ms(y), ms(y + 1)
        Ts = []
        for d in (1, -1):
            clf, (thr, vR, vn) = fit_and_thresholds(X, dec, lab, d, lo, H_ms)
            if thr is None:
                continue
            sel, auc = predict_trades(X, dec, lab, d, clf, thr, lo, hi)
            aucs.append((y, d, round(float(auc), 4)))
            Ts.append(trades_df(dec, lab, sel, d))
        if Ts:
            T = pd.concat(Ts).sort_values("dec_ms", kind="stable")
            T = T.iloc[greedy_idx(T["dec_ms"].values, T["exit_ts"].values, np.arange(len(T)))]
            allT.append(T)
        print(f"  cfg{cfg} year {y}: trades {0 if not Ts else len(T)}", flush=True)
    T = pd.concat(allT) if allT else pd.DataFrame(columns=["dec_ms", "exit_ts", "R", "net_pips", "dir", "filled"])
    return T, aucs


def conf_run(cfg):
    s, rr, H = cfg
    H_ms = H * 5 * MS_MIN
    Bd = block("disc", cfg)
    Bc = block("conf", cfg)
    X, dec, lab = concat_blocks([Bd, Bc])
    lo_disc_end, lo_c, hi_c = ms(2017), ms(2017), ms(2022, 7)
    Ts, aucs, thrs = [], [], {}
    for d in (1, -1):
        clf, (thr, vR, vn) = fit_and_thresholds(X, dec, lab, d, lo_disc_end, H_ms)
        thrs[d] = {"thr": thr, "val_meanR": vR, "val_n": vn}
        if thr is None:
            continue
        sel, auc = predict_trades(X, dec, lab, d, clf, thr, lo_c, hi_c)
        aucs.append((d, round(float(auc), 4)))
        Ts.append(trades_df(dec, lab, sel, d))
    T = pd.concat(Ts).sort_values("dec_ms", kind="stable")
    T = T.iloc[greedy_idx(T["dec_ms"].values, T["exit_ts"].values, np.arange(len(T)))] if len(T) else T
    return T, aucs, thrs


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "disc":
        rows = []
        cfg_list = [eval(sys.argv[2])] if len(sys.argv) > 2 else CFGS
        for cfg in cfg_list:
            T, aucs = wf_disc(cfg)
            s_ = summarize_trades(T, str(cfg))
            fr = approx_frontier(T) if len(T) >= 5 else {}
            print("DISC-WF", s_, {k: round(float(v), 4) for k, v in fr.items() if v == v}, "AUC(test,dir):", aucs, flush=True)
            rows.append({**s_, **fr})
            if len(T):
                T.to_parquet(f"/tmp/hipo_r2/results/ml_disc_{cfg[0]}_{cfg[1]}_{cfg[2]}.parquet")
        tag = "_".join(str(x) for x in cfg_list[0]) if len(sys.argv) > 2 else "all"
        pd.DataFrame(rows).to_csv(f"/tmp/hipo_r2/results/ml_disc_summary_{tag}.csv", index=False)
    else:
        cfg = eval(sys.argv[2])
        T, aucs, thrs = conf_run(cfg)
        s_ = summarize_trades(T, "CONF " + str(cfg))
        fr = approx_frontier(T) if len(T) >= 5 else {}
        print("CONF", s_, {k: round(float(v), 4) for k, v in fr.items() if v == v}, "AUC:", aucs, "thr:", thrs, flush=True)
        if len(T):
            T.to_parquet(f"/tmp/hipo_r2/results/ml_conf_{cfg[0]}_{cfg[1]}_{cfg[2]}.parquet")
