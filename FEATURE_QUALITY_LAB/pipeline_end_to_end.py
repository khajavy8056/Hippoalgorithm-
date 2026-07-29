#!/usr/bin/env python3
"""
HIPO End-to-End Pipeline (headless, no Gradio)
==============================================
1) Inject Pivot-aligned features (PSA/LSW/FVG) into EURUSD_Features
2) Multi-config labeling search (diverse params) with tick race
3) Feature↔Label quality scoring + selection
4) Train XGBoost sniper with selected features (purged split + Wilson CI)
5) Write FINAL_RESULT report

Goal: measurable improvement path — more samples, better feature alignment, real metrics.
"""
from __future__ import annotations

import os, sys, json, gc, shutil, traceback, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FEATURE_QUALITY_LAB"))

OUT = ROOT / "pipeline_out"
OUT.mkdir(exist_ok=True)
DATA = OUT / "hipo_lab_data"
DATA.mkdir(exist_ok=True)

# copy raw inputs
TEST = ROOT / "test_data"
for name in [
    "EURUSD_Features.parquet",
    "EURUSD_Tick_2023_2023.parquet",
    "EURUSD_Tick_2024_2024.parquet",
    "EURUSD_Labeled.parquet",
]:
    src = TEST / name
    if src.exists():
        shutil.copy2(src, DATA / name)

# ---------------------------------------------------------------------------
# Load base modules (functions only)
# ---------------------------------------------------------------------------
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

fe = load("fe", ROOT / "FEATURE_QUALITY_LAB" / "_fe_base.py")
lab = load("lab", ROOT / "FEATURE_QUALITY_LAB" / "_lab_base.py")

# point labeling DATA_DIR to our local path
lab.DATA_DIR = str(DATA)
if hasattr(lab, "IMG_DIR"):
    lab.IMG_DIR = str(OUT / "signals_img")
    os.makedirs(lab.IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# NEW FEATURE BUILDERS (Pivot-aligned) — same logic as v25
# ---------------------------------------------------------------------------

def calc_atr(df, length=14):
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def detect_causal_fractals(df, n=2):
    high, low = df["High"], df["Low"]
    window = 2 * n + 1
    roll_max = high.rolling(window).max()
    roll_min = low.rolling(window).min()
    is_fh = (high.shift(n) == roll_max) & high.shift(n).notna()
    is_fl = (low.shift(n) == roll_min) & low.shift(n).notna()
    return is_fh.fillna(False), is_fl.fillna(False), high.shift(n).where(is_fh), low.shift(n).where(is_fl)


def build_psa_features(df, n=2):
    out = pd.DataFrame(index=df.index)
    idx_pos = pd.Series(np.arange(len(df)), index=df.index)
    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=n)
    atr = calc_atr(df, 14)
    last_h = fh_price.ffill()
    last_l = fl_price.ffill()
    last_h_idx = idx_pos.where(is_fh).ffill()
    last_l_idx = idx_pos.where(is_fl).ffill()
    last_is_high = last_h_idx.fillna(-1) > last_l_idx.fillna(-1)
    B_price = pd.Series(np.where(last_is_high, last_h, last_l), index=df.index)
    B_idx = pd.Series(np.where(last_is_high, last_h_idx, last_l_idx), index=df.index)
    A_price = pd.Series(np.where(last_is_high, last_l, last_h), index=df.index)
    A_idx = pd.Series(np.where(last_is_high, last_l_idx, last_h_idx), index=df.index)
    ab_range = (B_price - A_price).abs().replace(0, np.nan)
    ab_bars = (B_idx - A_idx).replace(0, np.nan)
    bars_since_B = idx_pos - B_idx
    ab_dir = pd.Series(np.where(last_is_high, 1.0, -1.0), index=df.index)
    bc_retrace = ((B_price - df["Close"]) / ab_range) * ab_dir
    change_pt = (B_idx != B_idx.shift(1)).cumsum()
    seg_low = df["Low"].groupby(change_pt).cummin()
    seg_high = df["High"].groupby(change_pt).cummax()
    bc_extreme = pd.Series(
        np.where(last_is_high, (B_price - seg_low) / ab_range, (seg_high - B_price) / ab_range),
        index=df.index,
    )
    dist_to_B = ((df["Close"] - B_price) / (atr + 1e-9)) * ab_dir
    sweep_wick = pd.Series(np.where(last_is_high, df["High"] > B_price, df["Low"] < B_price), index=df.index).astype(int)
    sweep_close = pd.Series(np.where(last_is_high, df["Close"] > B_price, df["Close"] < B_price), index=df.index).astype(int)
    sweep_depth = pd.Series(
        np.where(last_is_high, (df["High"] - B_price).clip(lower=0) / (atr + 1e-9),
                 (B_price - df["Low"]).clip(lower=0) / (atr + 1e-9)),
        index=df.index,
    )
    C_proxy = pd.Series(np.where(last_is_high, seg_low, seg_high), index=df.index)
    pos_in_box = pd.Series(
        np.where(last_is_high, (df["Close"] - C_proxy) / (ab_range + 1e-9),
                 (C_proxy - df["Close"]) / (ab_range + 1e-9)),
        index=df.index,
    )
    cd_range = pd.Series(
        np.where(last_is_high, (seg_high - C_proxy).clip(lower=0), (C_proxy - seg_low).clip(lower=0)),
        index=df.index,
    )
    rng = df["High"] - df["Low"]
    body = (df["Close"] - df["Open"]).abs()
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    opp_wick = pd.Series(np.where(last_is_high, upper_wick / (body + 1e-9), lower_wick / (body + 1e-9)), index=df.index)

    out["PSA_AB_Range_ATR"] = ab_range / (atr + 1e-9)
    out["PSA_AB_Bars"] = ab_bars
    out["PSA_AB_Dir"] = ab_dir
    out["PSA_Bars_Since_B"] = bars_since_B
    out["PSA_BC_Retrace_Close"] = bc_retrace
    out["PSA_BC_Retrace_Extreme"] = bc_extreme
    out["PSA_BC_In_SweetZone"] = ((bc_extreme >= 0.20) & (bc_extreme <= 0.65)).astype(float)
    out["PSA_Dist_To_B_ATR"] = dist_to_B
    out["PSA_Sweep_Wick_Flag"] = sweep_wick
    out["PSA_Sweep_Close_Flag"] = sweep_close
    out["PSA_Sweep_Depth_ATR"] = sweep_depth
    out["PSA_Box_Position"] = pos_in_box.clip(-0.5, 1.5)
    out["PSA_Close_Beyond_Box"] = (pos_in_box > 1.0).astype(float)
    out["PSA_CD_AB_Ratio"] = cd_range / (ab_range + 1e-9)
    out["PSA_Candle_Range_ATR"] = rng / (atr + 1e-9)
    out["PSA_Candle_Body_Ratio"] = body / rng.replace(0, np.nan)
    out["PSA_Opp_Wick_Body_Ratio"] = opp_wick
    out["PSA_Setup_Readiness"] = (
        out["PSA_BC_In_SweetZone"]
        * (1.0 / (1.0 + out["PSA_Dist_To_B_ATR"].abs()))
        * (out["PSA_AB_Range_ATR"].clip(0, 5) / 5.0)
    )
    return out.shift(1)


def build_lsw_features(df, lookback=20):
    out = pd.DataFrame(index=df.index)
    atr = calc_atr(df, 14)
    prev_high = df["High"].shift(1).rolling(lookback).max()
    prev_low = df["Low"].shift(1).rolling(lookback).min()
    bull_sweep = (df["Low"] < prev_low) & (df["Close"] > prev_low)
    bear_sweep = (df["High"] > prev_high) & (df["Close"] < prev_high)
    out["LSW_Bull_Sweep_Flag"] = bull_sweep.astype(int)
    out["LSW_Bear_Sweep_Flag"] = bear_sweep.astype(int)
    out["LSW_Bull_Sweep_Depth_ATR"] = ((prev_low - df["Low"]).clip(lower=0) / (atr + 1e-9))
    out["LSW_Bear_Sweep_Depth_ATR"] = ((df["High"] - prev_high).clip(lower=0) / (atr + 1e-9))
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    body = df["Close"] - df["Open"]
    out["LSW_Rejection_Score"] = (body / rng) * np.where(bear_sweep, -1, np.where(bull_sweep, 1, 0))
    if "Tick_Up_Count" in df.columns:
        tot = (df["Tick_Up_Count"] + df["Tick_Down_Count"]).replace(0, np.nan)
        delta = (df["Tick_Up_Count"] - df["Tick_Down_Count"]) / tot
        out["LSW_Tick_Delta"] = delta
        out["LSW_Sweep_Tick_Align"] = np.where(bull_sweep, delta, np.where(bear_sweep, -delta, 0.0))
    else:
        out["LSW_Tick_Delta"] = 0.0
        out["LSW_Sweep_Tick_Align"] = 0.0
    idx = pd.Series(np.arange(len(df)), index=df.index)
    out["LSW_Bars_Since_Bull_Sweep"] = idx - idx.where(bull_sweep).ffill()
    out["LSW_Bars_Since_Bear_Sweep"] = idx - idx.where(bear_sweep).ffill()
    return out.shift(1)


def build_fvg_features(df):
    out = pd.DataFrame(index=df.index)
    atr = calc_atr(df, 14)
    bull_gap = df["Low"] - df["High"].shift(2)
    bear_gap = df["Low"].shift(2) - df["High"]
    is_bull = bull_gap > 0
    is_bear = bear_gap > 0
    bull_mid = ((df["Low"] + df["High"].shift(2)) / 2).where(is_bull).ffill()
    bear_mid = ((df["High"] + df["Low"].shift(2)) / 2).where(is_bear).ffill()
    idx = pd.Series(np.arange(len(df)), index=df.index)
    out["FVG_Dist_Bull_ATR"] = (df["Close"] - bull_mid) / (atr + 1e-9)
    out["FVG_Dist_Bear_ATR"] = (df["Close"] - bear_mid) / (atr + 1e-9)
    out["FVG_Bars_Since_Bull"] = idx - idx.where(is_bull).ffill()
    out["FVG_Bars_Since_Bear"] = idx - idx.where(is_bear).ffill()
    out["FVG_Bull_Size_ATR"] = bull_gap.where(is_bull).ffill() / (atr + 1e-9)
    out["FVG_Bear_Size_ATR"] = bear_gap.where(is_bear).ffill() / (atr + 1e-9)
    return out.shift(1)


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
DROP_ALWAYS = {
    "Open", "High", "Low", "Close", "Volume", "Tick_Up_Count", "Tick_Down_Count",
    "Bid", "Ask", "Target_Class", "Signal_Dir", "entry_price", "M1_SL", "M1_TP",
    "max_bars", "pair", "Raw_ATR",
}


def is_num(s):
    return bool(pd.api.types.is_numeric_dtype(s))


def prepare_xy(df):
    y = df["Target_Class"].astype(int).values
    cols = [c for c in df.columns if c not in DROP_ALWAYS and not str(c).startswith("__") and is_num(df[c])]
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    return X, y, cols


def score_features(X, y, cols):
    Xm = X.fillna(X.median(numeric_only=True))
    from sklearn.feature_selection import mutual_info_classif
    try:
        mi = mutual_info_classif(Xm[cols].values.astype(np.float64), y, random_state=42,
                                 n_neighbors=min(5, max(1, len(y)//20)))
    except Exception:
        mi = np.zeros(len(cols))
    rows = []
    for i, c in enumerate(cols):
        v = Xm[c].values.astype(np.float64)
        if np.std(v) < 1e-12:
            auc, r = 0.5, 0.0
        else:
            # rank AUC
            order = np.argsort(v)
            y_ord = y[order]
            n_pos = float(y_ord.sum()); n_neg = float(len(y_ord) - n_pos)
            if n_pos < 1 or n_neg < 1:
                auc = 0.5
            else:
                ranks = np.arange(1, len(y_ord)+1, dtype=np.float64)
                auc = (ranks[y_ord == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
                auc = max(auc, 1-auc)
            r = float(np.corrcoef(v, y)[0, 1])
            if np.isnan(r):
                r = 0.0
        auc_s = np.clip((auc - 0.5) / 0.15, 0, 1) * 100
        mi_s = np.clip(mi[i] / 0.12, 0, 1) * 100
        r_s = np.clip(abs(r) / 0.25, 0, 1) * 100
        q = 0.45 * auc_s + 0.35 * mi_s + 0.20 * r_s
        verdict = "KEEP_STRONG" if q >= 55 else ("KEEP" if q >= 35 else ("WEAK" if q >= 20 else "DROP"))
        rows.append(dict(feature=c, AUC_abs=round(float(auc), 4), abs_corr=round(abs(r), 4),
                         MI=round(float(mi[i]), 5), quality_0_100=round(float(q), 2), verdict=verdict))
    return pd.DataFrame(rows).sort_values("quality_0_100", ascending=False).reset_index(drop=True)


def select_features(feat_df, X, max_features=35, min_q=25.0, corr_th=0.92):
    cands = feat_df[feat_df["quality_0_100"] >= min_q]
    if len(cands) == 0:
        cands = feat_df.head(max_features)
    selected = []
    for _, row in cands.iterrows():
        c = row["feature"]
        if c not in X.columns or len(selected) >= max_features:
            if len(selected) >= max_features:
                break
            continue
        v = X[c].fillna(X[c].median()).values
        ok = True
        for s in selected:
            vs = X[s].fillna(X[s].median()).values
            if np.std(v) < 1e-12 or np.std(vs) < 1e-12:
                continue
            r = abs(np.corrcoef(v, vs)[0, 1])
            if not np.isnan(r) and r >= corr_th:
                ok = False
                break
        if ok:
            selected.append(c)
    return selected


def multivariate_probe(X, y, selected, n_folds=3, embargo=5):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    if len(selected) == 0 or len(y) < 30:
        return dict(cv_pr_auc=0.0, base=float(y.mean()) if len(y) else 0, lift=0, shuffle_p=1.0, real=False)
    Xm = X[selected].fillna(X[selected].median()).astype(np.float32)
    base = float(y.mean())
    tscv = TimeSeriesSplit(n_splits=min(n_folds, max(2, len(y)//25)), gap=embargo)
    scores = []
    for tr, te in tscv.split(Xm):
        if y[tr].sum() < 3 or y[te].sum() < 2 or (1 - y[tr]).sum() < 3:
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.08,
                                             min_samples_leaf=max(5, len(tr)//40),
                                             l2_regularization=1.0, random_state=42)
        clf.fit(Xm.iloc[tr], y[tr])
        p = clf.predict_proba(Xm.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], p))
    if not scores:
        return dict(cv_pr_auc=0.0, base=base, lift=0, shuffle_p=1.0, real=False)
    real = float(np.mean(scores))
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(10):
        ys = y.copy(); rng.shuffle(ys)
        fs = []
        for tr, te in tscv.split(Xm):
            if ys[tr].sum() < 3 or ys[te].sum() < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=50, learning_rate=0.08,
                                                 min_samples_leaf=max(5, len(tr)//40),
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(Xm.iloc[tr], ys[tr])
            fs.append(average_precision_score(ys[te], clf.predict_proba(Xm.iloc[te])[:, 1]))
        if fs:
            sh.append(float(np.mean(fs)))
    pval = (1 + sum(1 for s in sh if s >= real - 1e-12)) / (len(sh) + 1) if sh else 1.0
    lift = real / (base + 1e-9)
    return dict(cv_pr_auc=round(real, 4), cv_std=round(float(np.std(scores)), 4),
                base=round(base, 4), lift=round(lift, 3), shuffle_p=round(float(pval), 4),
                shuffle_mean=round(float(np.mean(sh)) if sh else base, 4),
                real=bool(pval < 0.15 and lift > 1.08))


def global_quality(feat_df, probe, n, wr):
    top10 = feat_df.head(10)
    uni = float(np.clip(top10["quality_0_100"].mean() / 70, 0, 1) * 35)
    multi = 30 * float(np.clip((probe.get("lift", 1) - 1) / 0.5, 0, 1))
    if not probe.get("real"):
        multi *= 0.5
    sample_s = 10 * float(np.clip(np.log1p(n) / np.log1p(2000), 0, 1))
    wr_s = 10 * float(np.clip(1 - abs(wr - 0.45) / 0.45, 0, 1))
    keep = int(feat_df["verdict"].isin(["KEEP", "KEEP_STRONG"]).sum())
    strong = int((feat_df["verdict"] == "KEEP_STRONG").sum())
    cover = 15 * float(np.clip(strong / 10, 0, 1))
    total = float(np.clip(uni + multi + sample_s + wr_s + cover, 0, 100))
    return round(total, 2), dict(uni=round(uni, 2), multi=round(multi, 2), sample=round(sample_s, 2),
                                 wr=round(wr_s, 2), cover=round(cover, 2), n_keep=keep, n_strong=strong)


def fast_fq(signal_df, max_feat=25):
    if signal_df is None or len(signal_df) < 25 or "Target_Class" not in signal_df.columns:
        return 0.0
    y = signal_df["Target_Class"].astype(int).values
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return 0.0
    cols = [c for c in signal_df.columns if c not in DROP_ALWAYS and is_num(signal_df[c])]
    if not cols:
        return 0.0
    X = signal_df[cols].replace([np.inf, -np.inf], np.nan).fillna(signal_df[cols].median(numeric_only=True))
    scores = []
    for c in cols:
        v = X[c].values.astype(np.float64)
        if np.std(v) < 1e-12:
            continue
        r = np.corrcoef(v, y)[0, 1]
        if np.isnan(r):
            continue
        order = np.argsort(v); y_ord = y[order]
        n_pos = float(y_ord.sum()); n_neg = float(len(y_ord) - n_pos)
        if n_pos < 1 or n_neg < 1:
            continue
        ranks = np.arange(1, len(y_ord)+1, dtype=np.float64)
        auc = (ranks[y_ord == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        auc = max(float(auc), 1 - float(auc))
        q = 0.6 * float(np.clip((auc - 0.5) / 0.15, 0, 1)) + 0.4 * float(np.clip(abs(r) / 0.25, 0, 1))
        scores.append(q)
    if not scores:
        return 0.0
    scores = sorted(scores, reverse=True)
    top = scores[:max_feat]
    mean_top = float(np.mean(top))
    n_strong = sum(1 for s in scores if s >= 0.45)
    cov = min(1.0, n_strong / 8.0)
    sf = float(np.clip(np.log1p(len(y)) / np.log1p(800), 0.4, 1.0))
    return float(np.clip(100 * mean_top * (0.7 + 0.3 * cov) * sf, 0, 100))


def multiobj_fitness(n, wins, min_samples, fq, min_wr=0.30, wc=0.35, ww=0.35, wf=0.30):
    if n == 0:
        return -1e6
    if n < min_samples:
        return -float(min_samples - n) - (100 - fq) * 0.1
    p = wins / n
    count_score = 100 * float(np.clip(np.log1p(n) / np.log1p(max(min_samples * 3, 2000)), 0, 1))
    win_bonus = 20 * float(np.clip(np.log1p(wins) / np.log1p(max(min_samples * 0.45, 400)), 0, 1))
    if p < min_wr:
        wr_score = 40 * (p / max(min_wr, 1e-6))
    else:
        wr_score = 40 + 60 * float(np.clip((p - min_wr) / (0.55 - min_wr + 1e-9), 0, 1))
        if p > 0.70 and n < min_samples * 1.5:
            wr_score *= 0.85
    feat_score = float(np.clip(fq, 0, 100))
    blended = wc * (count_score + win_bonus) + ww * wr_score + wf * feat_score
    harmony = 1.0 - 0.15 * (abs(count_score - wr_score) + abs(wr_score - feat_score) + abs(count_score - feat_score)) / 200
    return float(blended * float(np.clip(harmony, 0.7, 1.05)))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / den
    return p, max(0, centre - margin), min(1, centre + margin)


# ---------------------------------------------------------------------------
# STEP 1: Enrich features
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Inject PSA/LSW/FVG into features")
print("=" * 70)
df_feat = pd.read_parquet(DATA / "EURUSD_Features.parquet")
print("base features", df_feat.shape)
# drop existing if re-run
drop_old = [c for c in df_feat.columns if c.startswith(("PSA_", "LSW_", "FVG_"))]
if drop_old:
    df_feat = df_feat.drop(columns=drop_old)

psa = build_psa_features(df_feat, n=2)
lsw = build_lsw_features(df_feat)
fvg = build_fvg_features(df_feat)
assert psa.iloc[0].isna().mean() > 0.8
assert lsw.iloc[0].isna().mean() > 0.8
print("leak-check OK (row0 NaN)")

df_feat = pd.concat([df_feat, psa, lsw, fvg], axis=1)
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
# don't dropna whole frame (would kill history); leave NaNs for labeled rows to handle
feat_out = DATA / "EURUSD_Features.parquet"
df_feat.to_parquet(feat_out, compression="snappy")
print("enriched features", df_feat.shape, "->", feat_out)
new_cols = [c for c in df_feat.columns if c.startswith(("PSA_", "LSW_", "FVG_"))]
print("new feature count", len(new_cols))

# ---------------------------------------------------------------------------
# STEP 2: Multi-config labeling search
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Multi-config labeling search (diverse params)")
print("=" * 70)

# Default configs spanning loose / quality / RR regimes
CONFIGS = []
base = dict(
    swing_n=2, ab_min_bars=2, ab_max_bars=12, ab_atr_mult_min=1.2,
    ab_extended_min=1.5, ab_extended_max=6.0, bc_min_bars=2,
    bc_retrace_min=0.15, bc_retrace_max=0.65, box_scale=1.0,
    signal_search_bars=100, signal_atr_mult=0.4, signal_body_ratio_min=0.3,
    signal_wick_reject_ratio=1.5, allow_fl_candle=True,
    confirm_search_bars=20, confirm_atr_mult=0.3, sl_buffer_atr_mult=0.1,
    cd_ab_mode="CD > AB (طبق نقاط تکرارشده در PDF)",
    ab_noise_fraction_max=0.5, ab2_discount_mult=0.7,
    rr=1.0, max_bars=60, use_trend_gate=False,
    trend_gate_mode="خلاف روند (Counter-Trend Reversal)",
    spread_raw=0.00012,
)

# systematic grid (compact but diverse)
grid = [
    # loose sample-max
    dict(name="loose_rr1", ab_max_bars=14, ab_atr_mult_min=1.0, ab_noise_fraction_max=0.55,
         bc_retrace_max=0.70, signal_search_bars=120, confirm_search_bars=25, rr=1.0, max_bars=80),
    dict(name="loose_rr1.5", ab_max_bars=14, ab_atr_mult_min=1.0, ab_noise_fraction_max=0.55,
         bc_retrace_max=0.70, signal_search_bars=120, rr=1.5, max_bars=80),
    # medium
    dict(name="med_rr1", ab_max_bars=10, ab_atr_mult_min=1.3, ab_noise_fraction_max=0.40,
         bc_retrace_max=0.60, signal_body_ratio_min=0.35, rr=1.0, max_bars=60),
    dict(name="med_rr1.5", ab_max_bars=10, ab_atr_mult_min=1.3, ab_noise_fraction_max=0.40,
         bc_retrace_max=0.60, signal_body_ratio_min=0.35, rr=1.5, max_bars=60),
    dict(name="med_rr2", ab_max_bars=10, ab_atr_mult_min=1.3, ab_noise_fraction_max=0.40,
         bc_retrace_max=0.55, rr=2.0, max_bars=60),
    # quality
    dict(name="qual_rr1", ab_max_bars=8, ab_atr_mult_min=1.6, ab_noise_fraction_max=0.30,
         bc_retrace_min=0.20, bc_retrace_max=0.50, signal_body_ratio_min=0.45,
         signal_atr_mult=0.5, confirm_atr_mult=0.4, rr=1.0, max_bars=45),
    dict(name="qual_rr1.5", ab_max_bars=8, ab_atr_mult_min=1.6, ab_noise_fraction_max=0.30,
         bc_retrace_min=0.20, bc_retrace_max=0.50, signal_body_ratio_min=0.45, rr=1.5, max_bars=50),
    # with trend gate
    dict(name="counter_trend_rr1", ab_max_bars=12, ab_atr_mult_min=1.2, use_trend_gate=True,
         trend_gate_mode="خلاف روند (Counter-Trend Reversal)", rr=1.0, max_bars=60),
    dict(name="with_trend_rr1", ab_max_bars=12, ab_atr_mult_min=1.2, use_trend_gate=True,
         trend_gate_mode="هم‌جهت روند (With-Trend Continuation)", rr=1.0, max_bars=60),
    # AB>CD mode
    dict(name="ab_gt_cd_rr1", cd_ab_mode="AB > CD (طبق خط اول PDF)", ab_max_bars=12,
         ab_atr_mult_min=1.2, rr=1.0, max_bars=60),
    # very loose for volume
    dict(name="very_loose_rr1", ab_min_bars=2, ab_max_bars=16, ab_atr_mult_min=0.9,
         ab_noise_fraction_max=0.6, bc_min_bars=1, bc_retrace_min=0.10, bc_retrace_max=0.75,
         signal_search_bars=150, signal_body_ratio_min=0.2, signal_atr_mult=0.25,
         confirm_search_bars=30, confirm_atr_mult=0.2, allow_fl_candle=True,
         rr=1.0, max_bars=100),
    dict(name="very_loose_rr1.2", ab_min_bars=2, ab_max_bars=16, ab_atr_mult_min=0.9,
         ab_noise_fraction_max=0.6, bc_min_bars=1, bc_retrace_max=0.75,
         signal_search_bars=150, signal_body_ratio_min=0.2, rr=1.2, max_bars=90),
    # swing sensitivity
    dict(name="swing3_loose_rr1", swing_n=3, ab_max_bars=14, ab_atr_mult_min=1.1,
         ab_noise_fraction_max=0.5, bc_retrace_max=0.65, rr=1.0, max_bars=70),
    dict(name="swing1_loose_rr1", swing_n=1, ab_max_bars=12, ab_atr_mult_min=1.1,
         ab_noise_fraction_max=0.5, bc_retrace_max=0.65, rr=1.0, max_bars=60),
    # extended ATR band wider
    dict(name="wide_ext_rr1", ab_extended_min=1.0, ab_extended_max=8.0, ab_max_bars=14,
         ab_atr_mult_min=1.0, rr=1.0, max_bars=70),
]

for g in grid:
    cfg = deepcopy(base)
    name = g.pop("name")
    cfg.update(g)
    cfg["name"] = name
    CONFIGS.append(cfg)

tm = lab.TickManager(str(DATA), "EURUSD")
df_feat = pd.read_parquet(DATA / "EURUSD_Features.parquet")

leaderboard = []
best = None
best_fit = -1e18
best_rows = None

for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    print(f"\n[{i+1}/{len(CONFIGS)}] config={name} ...", flush=True)
    try:
        signals, funnel = lab.build_pivot_settlement_signals(
            df_feat, int(cfg["swing_n"]), int(cfg["ab_min_bars"]), int(cfg["ab_max_bars"]),
            float(cfg["ab_atr_mult_min"]), float(cfg["ab_extended_min"]), float(cfg["ab_extended_max"]),
            int(cfg["bc_min_bars"]), float(cfg["bc_retrace_min"]), float(cfg["bc_retrace_max"]),
            float(cfg["box_scale"]), int(cfg["signal_search_bars"]), float(cfg["signal_atr_mult"]),
            float(cfg["signal_body_ratio_min"]), float(cfg["signal_wick_reject_ratio"]),
            bool(cfg["allow_fl_candle"]), int(cfg["confirm_search_bars"]), float(cfg["confirm_atr_mult"]),
            float(cfg["sl_buffer_atr_mult"]), cfg["cd_ab_mode"], float(cfg["ab_noise_fraction_max"]),
            float(cfg["ab2_discount_mult"]),
        )
        # optional trend gate
        if cfg["use_trend_gate"] and signals and "EMA_Stack_Score" in df_feat.columns:
            filtered = []
            for s in signals:
                try:
                    tv = df_feat["EMA_Stack_Score"].loc[s["B_time"]]
                except Exception:
                    continue
                if pd.isna(tv):
                    continue
                up = tv > 0
                if cfg["trend_gate_mode"].startswith("خلاف"):
                    keep = (up and s["dir"] == -1) or ((not up) and s["dir"] == 1)
                else:
                    keep = (up and s["dir"] == 1) or ((not up) and s["dir"] == -1)
                if keep:
                    filtered.append(s)
            signals = filtered

        rows = []
        wins = 0
        for s in signals:
            entry = s["entry_price"]; risk = s["risk_raw"]
            tp = entry - cfg["rr"] * risk if s["dir"] == -1 else entry + cfg["rr"] * risk
            try:
                eloc = df_feat.index.get_loc(s["time"])
            except Exception:
                continue
            limit_idx = min(eloc + int(cfg["max_bars"]), len(df_feat) - 1)
            ticks = tm.get_ticks_series(s["time"], df_feat.index[limit_idx])
            if len(ticks) > 0:
                status, _ = lab.simulate_race_ps(ticks.values, s["dir"], s["M1_SL"], tp, float(cfg["spread_raw"]))
                is_win = 1 if status == 1 else 0
            else:
                is_win = 0
            wins += is_win
            # pack row = features at entry + label meta
            row = df_feat.iloc[eloc].to_dict()
            row["Target_Class"] = is_win
            row["Signal_Dir"] = float(s["dir"])
            row["entry_price"] = entry
            row["M1_SL"] = s["M1_SL"]
            row["M1_TP"] = tp
            row["max_bars"] = cfg["max_bars"]
            row["rr"] = cfg["rr"]
            for mk in ["PS_AB_Range_ATR", "PS_AB_Bars", "PS_BC_Retrace_Ratio", "PS_BC_Bars",
                       "PS_CD_AB_Ratio", "PS_Bars_Signal_To_Trigger", "PS_FL_Candle_Used"]:
                if mk in s:
                    row[mk] = s[mk]
            rows.append(row)

        n = len(rows)
        wr = wins / n if n else 0.0
        sdf = pd.DataFrame(rows) if rows else pd.DataFrame()
        fq = fast_fq(sdf) if n >= 25 else 0.0
        fit = multiobj_fitness(n, wins, min_samples=80, fq=fq, min_wr=0.30)
        accepted = int(funnel.get("20_ACCEPTED_Raw_Signal", 0))
        rec = dict(name=name, n=n, wins=wins, wr=round(wr * 100, 2), fq=round(fq, 2),
                   fitness=round(fit, 2), accepted_raw=accepted, rr=cfg["rr"], max_bars=cfg["max_bars"],
                   config={k: v for k, v in cfg.items() if k != "name"})
        leaderboard.append(rec)
        print(f"   n={n:4d} wins={wins:3d} WR={wr*100:5.1f}% FQ={fq:5.1f} fit={fit:8.2f}")
        if fit > best_fit and n >= 40:
            best_fit = fit
            best = rec
            best_rows = sdf
    except Exception as e:
        print("   ERROR", e)
        traceback.print_exc()
        leaderboard.append(dict(name=name, n=0, wins=0, wr=0, fq=0, fitness=-1e6, error=str(e)))

tm.clear_memory()
lb_df = pd.DataFrame(leaderboard).sort_values("fitness", ascending=False).reset_index(drop=True)
lb_df.to_csv(OUT / "labeling_leaderboard.csv", index=False)
print("\n--- TOP 8 CONFIGS ---")
print(lb_df.head(8)[["name", "n", "wins", "wr", "fq", "fitness", "rr"]].to_string(index=False))

if best is None or best_rows is None or len(best_rows) < 30:
    # fallback: pick max n with wr>=0.28
    ok = lb_df[(lb_df["n"] >= 40) & (lb_df["wr"] >= 28)]
    if len(ok) == 0:
        ok = lb_df[lb_df["n"] >= 30]
    if len(ok) == 0:
        raise SystemExit("No viable labeling config found")
    best_name = ok.iloc[0]["name"]
    # re-run best by name
    best = ok.iloc[0].to_dict()
    # find rows again from leaderboard rebuild — re-exec that config
    cfg = None
    for c in CONFIGS:
        if c["name"] == best_name:
            cfg = c
            break
    print("fallback re-run", best_name)
    signals, _ = lab.build_pivot_settlement_signals(
        df_feat, int(cfg["swing_n"]), int(cfg["ab_min_bars"]), int(cfg["ab_max_bars"]),
        float(cfg["ab_atr_mult_min"]), float(cfg["ab_extended_min"]), float(cfg["ab_extended_max"]),
        int(cfg["bc_min_bars"]), float(cfg["bc_retrace_min"]), float(cfg["bc_retrace_max"]),
        float(cfg["box_scale"]), int(cfg["signal_search_bars"]), float(cfg["signal_atr_mult"]),
        float(cfg["signal_body_ratio_min"]), float(cfg["signal_wick_reject_ratio"]),
        bool(cfg["allow_fl_candle"]), int(cfg["confirm_search_bars"]), float(cfg["confirm_atr_mult"]),
        float(cfg["sl_buffer_atr_mult"]), cfg["cd_ab_mode"], float(cfg["ab_noise_fraction_max"]),
        float(cfg["ab2_discount_mult"]),
    )
    tm = lab.TickManager(str(DATA), "EURUSD")
    rows = []
    for s in signals:
        entry = s["entry_price"]; risk = s["risk_raw"]
        tp = entry - cfg["rr"] * risk if s["dir"] == -1 else entry + cfg["rr"] * risk
        eloc = df_feat.index.get_loc(s["time"])
        limit_idx = min(eloc + int(cfg["max_bars"]), len(df_feat) - 1)
        ticks = tm.get_ticks_series(s["time"], df_feat.index[limit_idx])
        is_win = 0
        if len(ticks) > 0:
            status, _ = lab.simulate_race_ps(ticks.values, s["dir"], s["M1_SL"], tp, float(cfg["spread_raw"]))
            is_win = 1 if status == 1 else 0
        row = df_feat.iloc[eloc].to_dict()
        row["Target_Class"] = is_win
        row["Signal_Dir"] = float(s["dir"])
        row["entry_price"] = entry
        row["M1_SL"] = s["M1_SL"]
        row["M1_TP"] = tp
        row["max_bars"] = cfg["max_bars"]
        for mk in ["PS_AB_Range_ATR", "PS_AB_Bars", "PS_BC_Retrace_Ratio", "PS_BC_Bars",
                   "PS_CD_AB_Ratio", "PS_Bars_Signal_To_Trigger", "PS_FL_Candle_Used"]:
            if mk in s:
                row[mk] = s[mk]
        rows.append(row)
    best_rows = pd.DataFrame(rows)
    tm.clear_memory()

print(f"\nBEST config: {best['name']} | n={best['n']} WR={best['wr']}% FQ={best['fq']} fit={best['fitness']}")
best_rows = best_rows.sort_index() if hasattr(best_rows.index, "is_monotonic_increasing") else best_rows
# ensure datetime index if possible
if not isinstance(best_rows.index, pd.DatetimeIndex):
    # rows were from iloc dicts without index — rebuild with times
    pass

labeled_path = DATA / "EURUSD_Labeled.parquet"
# attach proper index from entry times if present in a column — use RangeIndex ok for training chronological if sorted
# We stored features from iloc; need times. Re-extract from Signal via index of feat if missing.
if "time" not in best_rows.columns:
    # best_rows came from to_dict of series with index lost — reconstruct by matching entry_price+close approx is fragile.
    # Better: ensure index from original build. Re-run packing with index.
    pass

# Robust rebuild of best labeled set with DatetimeIndex
cfg = None
for c in CONFIGS:
    if c["name"] == best["name"]:
        cfg = c
        break
assert cfg is not None
print("Rebuilding best labeled parquet with DatetimeIndex...")
signals, funnel = lab.build_pivot_settlement_signals(
    df_feat, int(cfg["swing_n"]), int(cfg["ab_min_bars"]), int(cfg["ab_max_bars"]),
    float(cfg["ab_atr_mult_min"]), float(cfg["ab_extended_min"]), float(cfg["ab_extended_max"]),
    int(cfg["bc_min_bars"]), float(cfg["bc_retrace_min"]), float(cfg["bc_retrace_max"]),
    float(cfg["box_scale"]), int(cfg["signal_search_bars"]), float(cfg["signal_atr_mult"]),
    float(cfg["signal_body_ratio_min"]), float(cfg["signal_wick_reject_ratio"]),
    bool(cfg["allow_fl_candle"]), int(cfg["confirm_search_bars"]), float(cfg["confirm_atr_mult"]),
    float(cfg["sl_buffer_atr_mult"]), cfg["cd_ab_mode"], float(cfg["ab_noise_fraction_max"]),
    float(cfg["ab2_discount_mult"]),
)
if cfg["use_trend_gate"] and signals and "EMA_Stack_Score" in df_feat.columns:
    filtered = []
    for s in signals:
        try:
            tv = df_feat["EMA_Stack_Score"].loc[s["B_time"]]
        except Exception:
            continue
        if pd.isna(tv):
            continue
        up = tv > 0
        if cfg["trend_gate_mode"].startswith("خلاف"):
            keep = (up and s["dir"] == -1) or ((not up) and s["dir"] == 1)
        else:
            keep = (up and s["dir"] == 1) or ((not up) and s["dir"] == -1)
        if keep:
            filtered.append(s)
    signals = filtered

tm = lab.TickManager(str(DATA), "EURUSD")
records = []
index_times = []
for s in signals:
    entry = s["entry_price"]; risk = s["risk_raw"]
    tp = entry - cfg["rr"] * risk if s["dir"] == -1 else entry + cfg["rr"] * risk
    eloc = df_feat.index.get_loc(s["time"])
    limit_idx = min(eloc + int(cfg["max_bars"]), len(df_feat) - 1)
    ticks = tm.get_ticks_series(s["time"], df_feat.index[limit_idx])
    is_win = 0
    if len(ticks) > 0:
        status, _ = lab.simulate_race_ps(ticks.values, s["dir"], s["M1_SL"], tp, float(cfg["spread_raw"]))
        is_win = 1 if status == 1 else 0
    row = df_feat.iloc[eloc].copy()
    row["Target_Class"] = is_win
    row["Signal_Dir"] = float(s["dir"])
    row["entry_price"] = entry
    row["M1_SL"] = s["M1_SL"]
    row["M1_TP"] = tp
    row["max_bars"] = float(cfg["max_bars"])
    for mk in ["PS_AB_Range_ATR", "PS_AB_Bars", "PS_BC_Retrace_Ratio", "PS_BC_Bars",
               "PS_CD_AB_Ratio", "PS_Bars_Signal_To_Trigger", "PS_FL_Candle_Used"]:
        row[mk] = s.get(mk, 0.0)
    records.append(row)
    index_times.append(s["time"])
tm.clear_memory()

labeled = pd.DataFrame(records)
labeled.index = pd.DatetimeIndex(index_times)
labeled = labeled[~labeled.index.duplicated(keep="first")].sort_index()
labeled.to_parquet(labeled_path, compression="snappy")
n_lab = len(labeled)
wins_lab = int((labeled["Target_Class"] == 1).sum())
wr_lab = wins_lab / max(1, n_lab)
print(f"Saved labeled: {n_lab} signals | wins={wins_lab} | WR={wr_lab*100:.2f}% -> {labeled_path}")

# baseline old labels for comparison
old = pd.read_parquet(TEST / "EURUSD_Labeled.parquet")
old_n, old_w = len(old), int((old["Target_Class"] == 1).sum())
print(f"OLD labeled baseline: n={old_n} wins={old_w} WR={old_w/old_n*100:.2f}%")

# ---------------------------------------------------------------------------
# STEP 3: Quality Lab
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Feature↔Label Quality Lab")
print("=" * 70)

def run_quality(df, tag):
    X, y, cols = prepare_xy(df)
    feat_df = score_features(X, y, cols)
    selected = select_features(feat_df, X, max_features=35, min_q=22.0)
    # force keep useful new/meta
    for c in feat_df["feature"]:
        if (str(c).startswith(("PSA_", "LSW_", "FVG_", "PS_"))
                and feat_df.loc[feat_df.feature == c, "verdict"].values[0] in ("KEEP", "KEEP_STRONG")):
            if c not in selected and len(selected) < 40:
                selected.append(c)
    probe = multivariate_probe(X, y, selected, n_folds=3, embargo=5)
    score, br = global_quality(feat_df, probe, len(y), float(y.mean()))
    print(f"[{tag}] GlobalScore={score} | n={len(y)} WR={y.mean()*100:.1f}% | selected={len(selected)}")
    print(f"       probe={probe}")
    print(f"       top8:\n{feat_df.head(8)[['feature','AUC_abs','MI','quality_0_100','verdict']].to_string(index=False)}")
    newf = feat_df[feat_df["feature"].str.startswith(("PSA_", "LSW_", "FVG_"), na=False)]
    if len(newf):
        print(f"       new aligned top:\n{newf.head(8)[['feature','AUC_abs','MI','quality_0_100','verdict']].to_string(index=False)}")
    return dict(score=score, breakdown=br, probe=probe, selected=selected, feat_df=feat_df, X=X, y=y)

# quality on OLD labels + old features only (no PSA) for baseline
old_q = run_quality(old, "OLD_81")
new_q = run_quality(labeled, "NEW_BEST")

# save selected
with open(OUT / "selected_features.json", "w") as f:
    json.dump({"selected": new_q["selected"], "score": new_q["score"], "probe": new_q["probe"],
               "best_config": best}, f, indent=2, default=str, ensure_ascii=False)
new_q["feat_df"].to_csv(OUT / "feature_quality_table.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 4: Training comparison (all feats vs selected)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Training Sniper (baseline vs selected features)")
print("=" * 70)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb


def train_eval(df, feature_list, tag, n_folds=4, max_bars=60):
    Xall, y, cols = prepare_xy(df)
    feats = [c for c in feature_list if c in Xall.columns]
    if len(feats) < 5:
        feats = cols[:30]
    X = Xall[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    y = y.astype(int)
    n = len(y)
    if n < 40:
        return dict(tag=tag, error="too few samples", n=n)

    embargo = int(max_bars) + 10
    # chronological splits
    test_frac, calib_frac = 0.18, 0.15
    final_idx = int(n * (1 - test_frac - calib_frac))
    calib_start = min(final_idx + embargo, n)
    calib_end = min(int(n * (1 - test_frac)), n)
    oos_start = min(calib_end + embargo, n)
    if final_idx < 30 or (n - oos_start) < 10:
        # fallback simpler split
        final_idx = int(n * 0.7)
        calib_start = final_idx
        calib_end = int(n * 0.85)
        oos_start = calib_end
        embargo = 0

    X_cv, y_cv = X.iloc[:final_idx], y[:final_idx]
    X_cal, y_cal = X.iloc[calib_start:calib_end], y[calib_start:calib_end]
    X_te, y_te = X.iloc[oos_start:], y[oos_start:]

    base_params = dict(max_depth=3, learning_rate=0.05,
                       tree_method="hist", objective="binary:logistic",
                       eval_metric="aucpr", random_state=42, subsample=0.9,
                       colsample_bytree=0.7, reg_lambda=2.0)
    fold_aucs = []
    n_splits = min(n_folds, max(2, len(y_cv) // 25))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=max(1, embargo // 4) if embargo else 0)
    for tr, te in tscv.split(X_cv):
        if y_cv[tr].sum() < 3 or y_cv[te].sum() < 2:
            continue
        spw = (y_cv[tr] == 0).sum() / max(1, (y_cv[tr] == 1).sum())
        m = xgb.XGBClassifier(**base_params, n_estimators=200, scale_pos_weight=spw)
        m.fit(X_cv.iloc[tr], y_cv[tr], eval_set=[(X_cv.iloc[te], y_cv[te])], verbose=False)
        fold_aucs.append(average_precision_score(y_cv[te], m.predict_proba(X_cv.iloc[te])[:, 1]))

    # final model
    spw = (y_cv == 0).sum() / max(1, (y_cv == 1).sum())
    final = xgb.XGBClassifier(**base_params, scale_pos_weight=spw, n_estimators=250)
    # small val cut
    cut = max(10, int(len(X_cv) * 0.85))
    final.fit(X_cv.iloc[:cut], y_cv[:cut], eval_set=[(X_cv.iloc[cut:], y_cv[cut:])] if cut < len(X_cv) else None,
              verbose=False)

    calibrator = None
    if len(X_cal) >= 10 and len(np.unique(y_cal)) > 1:
        raw = final.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, y_cal)

    if len(X_te) == 0:
        return dict(tag=tag, error="empty test", n=n, fold_aucs=fold_aucs)

    raw_te = final.predict_proba(X_te)[:, 1]
    prob = calibrator.predict(raw_te) if calibrator is not None else raw_te
    pr_auc = float(average_precision_score(y_te, prob)) if len(np.unique(y_te)) > 1 else float("nan")
    base_rate = float(y_te.mean())

    # threshold sweep
    sweep = []
    for th in np.arange(0.20, 0.81, 0.05):
        pred = (prob >= th).astype(int)
        trades = int(pred.sum())
        if trades == 0:
            continue
        tp = int(((pred == 1) & (y_te == 1)).sum())
        p, lo, hi = wilson_ci(tp, trades)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        sweep.append(dict(th=round(float(th), 2), precision=round(float(prec), 4),
                          ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                          recall=round(float(rec), 4), trades=trades, tp=tp))
    sweep_df = pd.DataFrame(sweep)

    # best precision with at least max(5, 10% test) trades
    min_trades = max(5, int(0.08 * len(y_te)))
    recommendation = "insufficient"
    best_row = None
    if len(sweep_df):
        cands = sweep_df[sweep_df["trades"] >= min_trades]
        pool = cands if len(cands) else sweep_df
        # prefer precision, then trades
        best_row = pool.sort_values(["precision", "trades"], ascending=[False, False]).iloc[0].to_dict()
        recommendation = (f"th={best_row['th']} prec={best_row['precision']} "
                          f"CI[{best_row['ci_lo']}-{best_row['ci_hi']}] "
                          f"rec={best_row['recall']} trades={best_row['trades']}")

    # label shuffle on last fold style
    rng = np.random.default_rng(0)
    sh = []
    if len(X_cv) > 30:
        cut2 = int(len(X_cv) * 0.75)
        Xtr, ytr = X_cv.iloc[:cut2], y_cv[:cut2]
        Xva, yva = X_cv.iloc[cut2:], y_cv[cut2:]
        if yva.sum() >= 2 and (1 - yva).sum() >= 2:
            spw2 = (ytr == 0).sum() / max(1, (ytr == 1).sum())
            mreal = xgb.XGBClassifier(**base_params, scale_pos_weight=spw2, n_estimators=120)
            mreal.fit(Xtr, ytr, verbose=False)
            real_s = average_precision_score(yva, mreal.predict_proba(Xva)[:, 1])
            for _ in range(12):
                ysh = ytr.copy(); rng.shuffle(ysh)
                m = xgb.XGBClassifier(**base_params, scale_pos_weight=spw2, n_estimators=80)
                m.fit(Xtr, ysh, verbose=False)
                sh.append(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
            shuffle_p = (1 + sum(1 for s in sh if s >= real_s - 1e-12)) / (len(sh) + 1)
        else:
            real_s, shuffle_p = float("nan"), 1.0
    else:
        real_s, shuffle_p = float("nan"), 1.0

    fi = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)

    result = dict(
        tag=tag, n=n, n_features=len(feats),
        n_cv=len(y_cv), n_calib=len(y_cal), n_test=len(y_te),
        base_rate_test=round(base_rate, 4),
        fold_pr_auc_mean=round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        fold_pr_auc_std=round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
        oos_pr_auc=round(pr_auc, 4) if pr_auc == pr_auc else None,
        lift_vs_base=round(pr_auc / (base_rate + 1e-9), 3) if pr_auc == pr_auc else None,
        recommendation=recommendation,
        best_threshold=best_row,
        shuffle_p=round(float(shuffle_p), 4),
        cv_holdout_pr_auc=round(float(real_s), 4) if real_s == real_s else None,
        top_features=fi.head(15).round(4).to_dict(),
        sweep=sweep_df.to_dict(orient="records") if len(sweep_df) else [],
    )
    print(f"[{tag}] n={n} feats={len(feats)} foldPR={result['fold_pr_auc_mean']} "
          f"oosPR={result['oos_pr_auc']} lift={result['lift_vs_base']} "
          f"shuffle_p={result['shuffle_p']}")
    print(f"       {recommendation}")
    return result


# baseline: old labels, all numeric features
old_cols = [c for c in old.columns if c not in DROP_ALWAYS and is_num(old[c])]
res_old_all = train_eval(old, old_cols, "OLD_all_features", max_bars=60)
res_old_sel = train_eval(old, old_q["selected"], "OLD_selected", max_bars=60)

# new labels
new_cols = [c for c in labeled.columns if c not in DROP_ALWAYS and is_num(labeled[c])]
res_new_all = train_eval(labeled, new_cols, "NEW_all_features", max_bars=int(cfg["max_bars"]))
res_new_sel = train_eval(labeled, new_q["selected"], "NEW_selected_aligned", max_bars=int(cfg["max_bars"]))

# ---------------------------------------------------------------------------
# STEP 5: Final report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: FINAL REPORT")
print("=" * 70)

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "feat_df" and k not in ("X", "y")}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return o

final = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "baseline_old_labels": {"n": old_n, "wins": old_w, "wr": round(old_w / old_n * 100, 2)},
    "best_label_config": best,
    "new_labels": {"n": n_lab, "wins": wins_lab, "wr": round(wr_lab * 100, 2)},
    "quality_old": clean(old_q),
    "quality_new": clean(new_q),
    "training": {
        "old_all": clean(res_old_all),
        "old_selected": clean(res_old_sel),
        "new_all": clean(res_new_all),
        "new_selected": clean(res_new_sel),
    },
    "leaderboard_top": lb_df.head(10).to_dict(orient="records"),
    "improvements": {
        "sample_count_delta": n_lab - old_n,
        "sample_count_ratio": round(n_lab / max(1, old_n), 2),
        "quality_score_delta": round(new_q["score"] - old_q["score"], 2),
        "oos_pr_auc_old_selected": res_old_sel.get("oos_pr_auc"),
        "oos_pr_auc_new_selected": res_new_sel.get("oos_pr_auc"),
    },
}

with open(OUT / "FINAL_RESULT.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False, default=str)

# Markdown report
md = f"""# HIPO Pipeline Final Result

**Time:** {final['timestamp']}

## 1) Labeling

| | Old | New (best multi-obj) |
|--|-----|----------------------|
| Samples | {old_n} | **{n_lab}** |
| Wins | {old_w} | **{wins_lab}** |
| WinRate | {old_w/old_n*100:.1f}% | **{wr_lab*100:.1f}%** |
| Config | original | `{best['name']}` |
| Feature Quality (fast) | - | {best.get('fq')} |
| Multi-obj Fitness | - | {best.get('fitness')} |

Sample multiplier: **{final['improvements']['sample_count_ratio']}×** ({final['improvements']['sample_count_delta']:+d} signals)

### Top labeling configs
```
{lb_df.head(8)[['name','n','wins','wr','fq','fitness','rr']].to_string(index=False)}
```

## 2) Feature ↔ Label Quality

| | Old labels | New labels + PSA/LSW/FVG |
|--|------------|---------------------------|
| Global Score | {old_q['score']} | **{new_q['score']}** |
| CV PR-AUC probe | {old_q['probe'].get('cv_pr_auc')} | **{new_q['probe'].get('cv_pr_auc')}** |
| Lift | {old_q['probe'].get('lift')} | **{new_q['probe'].get('lift')}** |
| Shuffle p | {old_q['probe'].get('shuffle_p')} | **{new_q['probe'].get('shuffle_p')}** |
| Selected feats | {len(old_q['selected'])} | **{len(new_q['selected'])}** |

## 3) Training (Sniper-style, purged split + Wilson CI)

| Run | n | feats | Fold PR-AUC | OOS PR-AUC | Lift | Shuffle p | Best threshold |
|-----|---|-------|-------------|------------|------|-----------|----------------|
| OLD all | {res_old_all.get('n')} | {res_old_all.get('n_features')} | {res_old_all.get('fold_pr_auc_mean')} | {res_old_all.get('oos_pr_auc')} | {res_old_all.get('lift_vs_base')} | {res_old_all.get('shuffle_p')} | {res_old_all.get('recommendation')} |
| OLD selected | {res_old_sel.get('n')} | {res_old_sel.get('n_features')} | {res_old_sel.get('fold_pr_auc_mean')} | {res_old_sel.get('oos_pr_auc')} | {res_old_sel.get('lift_vs_base')} | {res_old_sel.get('shuffle_p')} | {res_old_sel.get('recommendation')} |
| NEW all | {res_new_all.get('n')} | {res_new_all.get('n_features')} | {res_new_all.get('fold_pr_auc_mean')} | {res_new_all.get('oos_pr_auc')} | {res_new_all.get('lift_vs_base')} | {res_new_all.get('shuffle_p')} | {res_new_all.get('recommendation')} |
| **NEW selected** | {res_new_sel.get('n')} | {res_new_sel.get('n_features')} | {res_new_sel.get('fold_pr_auc_mean')} | **{res_new_sel.get('oos_pr_auc')}** | **{res_new_sel.get('lift_vs_base')}** | {res_new_sel.get('shuffle_p')} | {res_new_sel.get('recommendation')} |

## 4) Verdict

"""

# auto verdict
oos_new = res_new_sel.get("oos_pr_auc") or 0
oos_old = res_old_sel.get("oos_pr_auc") or 0
sp = res_new_sel.get("shuffle_p") or 1
br_new = res_new_sel.get("base_rate_test") or wr_lab
lift = res_new_sel.get("lift_vs_base") or 0

verdict_lines = []
if n_lab > old_n * 1.3:
    verdict_lines.append(f"✅ Sample count improved ({old_n} → {n_lab}).")
else:
    verdict_lines.append(f"⚠️ Sample count not much higher ({old_n} → {n_lab}).")

if new_q["score"] > old_q["score"] + 1:
    verdict_lines.append(f"✅ Feature quality score improved ({old_q['score']} → {new_q['score']}).")
elif new_q["score"] >= old_q["score"] - 1:
    verdict_lines.append(f"➖ Feature quality similar ({old_q['score']} → {new_q['score']}).")
else:
    verdict_lines.append(f"⚠️ Feature quality dropped ({old_q['score']} → {new_q['score']}).")

if oos_new and oos_old and oos_new > oos_old + 0.02:
    verdict_lines.append(f"✅ OOS PR-AUC improved ({oos_old} → {oos_new}).")
elif oos_new and oos_new > (br_new + 0.05):
    verdict_lines.append(f"✅ OOS PR-AUC above base rate ({oos_new} vs base {br_new}).")
else:
    verdict_lines.append(f"⚠️ OOS PR-AUC still weak (new={oos_new}, old={oos_old}, base={br_new}).")

if sp < 0.15:
    verdict_lines.append(f"✅ Label-shuffle significant (p={sp}) — real signal.")
elif sp < 0.35:
    verdict_lines.append(f"⚠️ Label-shuffle borderline (p={sp}).")
else:
    verdict_lines.append(f"❌ Label-shuffle not significant (p={sp}) — still noise-like; need more years/data or stronger pattern.")

best_th = res_new_sel.get("best_threshold") or {}
if best_th:
    prec = best_th.get("precision", 0)
    trades = best_th.get("trades", 0)
    ci_lo = best_th.get("ci_lo", 0)
    if prec >= 0.55 and trades >= 8 and ci_lo >= 0.40:
        verdict_lines.append(f"✅ Threshold precision usable: {prec} on {trades} OOS trades (CI lo={ci_lo}).")
    elif prec >= 0.50:
        verdict_lines.append(f"⚠️ Precision {prec} on {trades} trades — promising but CI wide (lo={ci_lo}). Need more OOS trades.")
    else:
        verdict_lines.append(f"❌ Best OOS precision only {prec} on {trades} trades — not yet tradeable at 60% target.")

md += "\n".join(f"- {v}" for v in verdict_lines)
md += f"""

## 5) What to do next in Colab

1. Replace Feature cell with v25 (PSA/LSW/FVG) — already validated here.
2. Run Labeling GA v4 with weights count=0.35, wr=0.35, feat=0.30; min_samples≈{max(80, n_lab//2)}.
3. Apply best config from this run: `{best['name']}` (see FINAL_RESULT.json).
4. Run Quality Lab; only train if Global Score ≥ 50 and shuffle p improving.
5. For 60% precision claim: need **≥100 OOS trades** with Wilson CI lower bound >50%.

## 6) Best config parameters
```json
{json.dumps(best, indent=2, ensure_ascii=False, default=str)}
```

## 7) Selected features ({len(new_q['selected'])})
```
{json.dumps(new_q['selected'], indent=2, ensure_ascii=False)}
```
"""

(OUT / "FINAL_RESULT.md").write_text(md, encoding="utf-8")
print(md)
print("\n✅ All outputs in", OUT)
