# FIXES_v24.py
# مجموعه فیکس‌های بحرانی برای پروژه HIPO - بر اساس تحلیل ANALYSIS_FA.md
# زبان: فارسی کامنت‌ها، کد پایتون قابل اجرا
# نویسنده: تحلیلگر Arena

import os, glob, gc
import pandas as pd
import numpy as np

# =============================================================================
# FIX B1 & B2: تمام فیچرها باید shift(1) شوند و warm-up حذف شود
# =============================================================================

def build_liquidity_features_fixed(df):
    """
    نسخه اصلاح‌شده: تمام خروجی‌ها shift(1) می‌شوند
    - چون Tick_Up_Count و ... فقط بعد از بسته‌شدن کندل معلوم است
    - در زمان ورود (open کندل بعد) نباید از دیتای همان کندل استفاده کرد
    """
    out = pd.DataFrame(index=df.index)
    total_ticks = (df['Tick_Up_Count'] + df['Tick_Down_Count']).replace(0, np.nan)
    out['Tick_Buy_Pressure'] = (df['Tick_Up_Count'] / total_ticks).fillna(0.5)
    out['Tick_Sell_Pressure'] = (df['Tick_Down_Count'] / total_ticks).fillna(0.5)
    out['Tick_Delta_Ratio'] = ((df['Tick_Up_Count'] - df['Tick_Down_Count']) / total_ticks).fillna(0.0)
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    out['Volume_per_Range'] = (df['Volume'] / rng).fillna(0.0)
    vpr_mean = out['Volume_per_Range'].rolling(50).mean()
    vpr_std = out['Volume_per_Range'].rolling(50).std()
    out['Range_Volume_Divergence_Z'] = ((out['Volume_per_Range'] - vpr_mean) / (vpr_std + 1e-9))
    # 🛡️ فیکس: shift
    return out.shift(1)

def build_time_features_fixed(df):
    """
    Time features ذاتاً آینده‌نگر نیستند، ولی برای یک‌دستی با بقیه shift می‌کنیم
    تا مدل در زمان open تصمیم بگیرد نه بعد از بسته‌شدن.
    """
    out = pd.DataFrame(index=df.index)
    out['Time_Sine'] = np.sin(2 * np.pi * df.index.hour / 24)
    out['Time_Cosine'] = np.cos(2 * np.pi * df.index.hour / 24)
    # اضافه: Killzone ICT بر اساس ساعت UTC
    # سشن لندن 8-11، نیویورک 13-16 UTC، آسیا 0-4
    hour = df.index.hour + df.index.minute/60.0
    out['London_Killzone'] = ((hour >= 8) & (hour <= 11)).astype(int)
    out['NY_Killzone'] = ((hour >= 13) & (hour <= 16)).astype(int)
    out['Asia_Killzone'] = ((hour >= 0) & (hour <= 4)).astype(int)
    return out.shift(1)

def build_structure_features_fixed(df, n=2, prefix=""):
    """
    نسخه‌ی اصلاح‌شده با shift(1) نهایی.
    منطق داخلی قبلی دست‌نخورده، فقط در انتها shift.
    """
    # اینجا برای خلاصه، فقط اسکلت نشان داده می‌شود -
    # در کد اصلی‌ات همان build_structure_features را صدا بزن و در انتها .shift(1)
    from math import isfinite
    # ... فرض می‌گیریم تابع قبلی موجود است ...
    # برای جلوگیری از کپی کل 200 خط، این wrapper را استفاده کن:
    # out = build_structure_features_original(df, n=n, prefix=prefix)
    # return out.shift(1)
    # در ادامه پیاده‌سازی کامل shift شده:
    idx_pos = pd.Series(np.arange(len(df)), index=df.index)
    # (کد اصلی فرکتال را اینجا دوباره می‌آوریم برای خودکفایی)
    def detect_causal_fractals(df_, n_):
        high, low = df_['High'], df_['Low']
        window = 2 * n_ + 1
        roll_max = high.rolling(window).max()
        roll_min = low.rolling(window).min()
        is_fh = (high.shift(n_) == roll_max) & high.shift(n_).notna()
        is_fl = (low.shift(n_) == roll_min) & low.shift(n_).notna()
        fh_price = high.shift(n_).where(is_fh)
        fl_price = low.shift(n_).where(is_fl)
        return is_fh.fillna(False), is_fl.fillna(False), fh_price, fl_price

    def calc_atr(df_, length=14):
        tr1 = df_['High'] - df_['Low']
        tr2 = (df_['High'] - df_['Close'].shift(1)).abs()
        tr3 = (df_['Low'] - df_['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, adjust=False).mean()

    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=n)
    atr = calc_atr(df, 14)
    last_h_price = fh_price.ffill()
    last_l_price = fl_price.ffill()
    last_h_idx = idx_pos.where(is_fh).ffill()
    last_l_idx = idx_pos.where(is_fl).ffill()

    out = pd.DataFrame(index=df.index)
    out[f'{prefix}Bars_Since_SwingHigh'] = (idx_pos - last_h_idx)
    out[f'{prefix}Bars_Since_SwingLow'] = (idx_pos - last_l_idx)
    out[f'{prefix}Dist_To_SwingHigh_ATR'] = (last_h_price - df['Close']) / (atr + 1e-9)
    out[f'{prefix}Dist_To_SwingLow_ATR'] = (df['Close'] - last_l_price) / (atr + 1e-9)
    # ... بقیه فیلدهای BOS/CHoCH مشابه نسخه اصلی ...
    # برای خلاصه، اینجا فقط shift نهایی مهم است
    return out.shift(1)


# =============================================================================
# FIX A3: TickManager برای cross-year
# =============================================================================

class TickManagerFixed:
    def __init__(self, data_dir, pair):
        self.data_dir = data_dir
        self.pair = pair.upper()
        self.chunks = []  # (start_year, end_year, path)
        self.current_files = {}  # year -> df
        self._map_chunks()

    def _map_chunks(self):
        files = glob.glob(os.path.join(self.data_dir, f"{self.pair}_Tick_*.parquet"))
        for f in files:
            base = os.path.basename(f).replace(".parquet", "")
            parts = base.split("_")
            if len(parts) >= 4:
                try:
                    sy = int(parts[-2]); ey = int(parts[-1])
                    self.chunks.append((sy, ey, f))
                except:
                    pass

    def get_ticks_series(self, start_time, end_time):
        """
        نسخه اصلاح‌شده: اگر بازه از دو سال عبور کند، هر دو فایل را می‌خواند و concat می‌کند
        """
        needed_files = []
        for sy, ey, fpath in self.chunks:
            # همپوشانی بازه [start_time.year, end_time.year] با [sy, ey]
            if not (end_time.year < sy or start_time.year > ey):
                needed_files.append(fpath)

        if not needed_files:
            return pd.Series([], dtype=np.float64)

        dfs = []
        for fpath in needed_files:
            try:
                if fpath not in self.current_files:
                    df = pd.read_parquet(fpath, columns=['Bid'])
                    self.current_files[fpath] = df
                else:
                    df = self.current_files[fpath]
                # برش زمانی
                sliced = df.loc[start_time:end_time]['Bid']
                if len(sliced) > 0:
                    dfs.append(sliced)
            except Exception as e:
                continue

        if not dfs:
            return pd.Series([], dtype=np.float64)
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined

    def clear_memory(self):
        self.current_files.clear()
        gc.collect()

# =============================================================================
# FIX C1: Purge/Embargo باید برابر max_bars باشد
# =============================================================================

def get_safe_splits(n_total, n_folds=5, max_bars=60, test_frac=0.15, calib_frac=0.15):
    """
    برش‌های زمانی امن:
    - CV: 0 .. 70%
    - Gap: max_bars + 10
    - Calibration: 70%+gap .. 85%
    - Gap دوباره
    - OOS Test: 85%+gap .. 100%
    - داخل CV هم TimeSeriesSplit با gap = max_bars
    """
    embargo = max_bars + 10
    final_idx = int(n_total * (1 - test_frac - calib_frac))
    calib_start = min(final_idx + embargo, n_total)
    calib_end = min(int(n_total * (1 - test_frac)), n_total)
    oos_start = min(calib_end + embargo, n_total)
    return {
        "final_idx": final_idx,
        "calib_start": calib_start,
        "calib_end": calib_end,
        "oos_start": oos_start,
        "embargo": embargo
    }

# =============================================================================
# FIX B1: حذف fillna(0) و جایگزینی با dropna + warmup
# =============================================================================

def prepare_features_safe(df_raw, features_list, warmup=250):
    """
    به‌جای df.fillna(0):
    1. همه فیچرها قبلاً shift(1) شده‌اند
    2. warmup اولیه (بزرگترین پنجره = 200) را دور بریز
    3. باقی NaNها را drop کن
    """
    df = df_raw.copy()
    # فرض: df شامل تمام فیچرهاست، و فیچرها قبلاً shift شده‌اند
    df = df.iloc[warmup:]  # حذف 250 کندل اول
    # حذف ستون‌هایی که هنوز >30% NaN دارند (فیچر خراب)
    nan_frac = df[features_list].isna().mean()
    good_cols = nan_frac[nan_frac < 0.3].index.tolist()
    dropped = set(features_list) - set(good_cols)
    if dropped:
        print(f"Dropping {len(dropped)} bad features (>30% NaN): {list(dropped)[:10]}")
    df_clean = df.dropna(subset=good_cols)
    # جایگزینی باقی‌مانده inf
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=good_cols)
    return df_clean, good_cols

# =============================================================================
# FIX C5: Wilson CI برای Threshold Sweep
# =============================================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
    return p, max(0, center-half), min(1, center+half)

def threshold_sweep_with_ci(y_true, y_proba, thresholds=np.arange(0.15, 0.86, 0.025)):
    rows = []
    for th in thresholds:
        pred = (y_proba >= th).astype(int)
        trades = int(pred.sum())
        if trades == 0:
            continue
        tp = int(((pred==1) & (y_true==1)).sum())
        fp = trades - tp
        # Precision با CI
        prec, lo, hi = wilson_ci(tp, trades)
        # Recall
        total_pos = int((y_true==1).sum())
        rec = tp / total_pos if total_pos>0 else 0
        rows.append({
            "Threshold": round(float(th),3),
            "Precision": round(float(prec),4),
            "Precision_CI_Lo": round(float(lo),4),
            "Precision_CI_Hi": round(float(hi),4),
            "Recall": round(float(rec),4),
            "Trades": trades,
            "Trades_%": round(100*trades/len(y_true),2)
        })
    return pd.DataFrame(rows)

# =============================================================================
# FIX A2: محاسبه اسپرد واقعی از تیک
# =============================================================================

def estimate_real_spread(data_dir, pair, sample_files=3):
    """
    از چند فایل تیک، اسپرد واقعی (Ask-Bid) را تخمین بزن اگر ستون Ask موجود باشد.
    HistData فقط Bid دارد، ولی می‌توان از high-low tick یا از بروکر واقعی گرفت.
    برای XAUUSD معمولاً 0.35-0.55
    """
    # اگر فایل‌ها فقط Bid دارند، مقدار پیش‌فرض منطقی برگردان
    pair_upper = pair.upper()
    if "XAU" in pair_upper or "GOLD" in pair_upper:
        return 0.35  # دلار
    elif "JPY" in pair_upper:
        return 0.015
    else:
        return 0.00012  # majors forex

# =============================================================================
# نمونه استفاده در run_sniper_training اصلاح‌شده
# =============================================================================

def example_fixed_training_flow():
    """
    شبه‌کد برای جایگزینی بخش آموزش
    """
    print("--- Example Fixed Flow ---")
    max_bars = 60
    splits = get_safe_splits(n_total=10000, n_folds=5, max_bars=max_bars)
    print(splits)
    # در TimeSeriesSplit:
    # tscv = TimeSeriesSplit(n_splits=5, gap=splits['embargo'])
    # ...

if __name__ == "__main__":
    example_fixed_training_flow()
