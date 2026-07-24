# @title 🧠 HIPO STRUCTURE ENGINE [v24.0 FIXED - Zero Leak + Killzone + Warmup Drop] { display-mode: "form" }
# =============================================================================
# تغییرات کلیدی نسبت به v23:
# 1. همه build_* ها در انتها .shift(1) می‌شوند → ورود در open کندل بعد، بدون دیدن close همان کندل
# 2. build_liquidity_features که قبلا بدون shift بود الان FIXED شد
# 3. build_time_features + Killzone لندن/نیویورک/آسیا اضافه شد + shift
# 4. build_structure_features در انتها shift(1) → BOS/CHoCH دیگر از close همان کندل entry استفاده نمی‌کند
# 5. پس از concat تمام بلوک‌ها، 250 کندل اول (warmup بزرگترین پنجره 200) + تمام NaN ها drop می‌شوند
#    به‌جای fillna(0) که آلودگی می‌ساخت
# 6. گزارش NaN fraction قبل از ذخیره
# =============================================================================

import sys, subprocess
def smart_install():
    reqs = ["gradio", "pyarrow", "numba"]
    missing = []
    for req in reqs:
        try:
            __import__(req)
        except ImportError:
            missing.append(req)
    if missing:
        print(f"⏳ Installing Infrastructure ({', '.join(missing)})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        from IPython.display import clear_output
        clear_output()
smart_install()

import os, glob, gc, traceback
import pandas as pd
import numpy as np
import gradio as gr
from numba import njit

DATA_DIR = "/content/hipo_lab_data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def get_available_pairs():
    files = glob.glob(os.path.join(DATA_DIR, "*_Tick*.parquet"))
    pairs = set(os.path.basename(f).split('_')[0] for f in files)
    return list(pairs) if pairs else ["No Tick Data Found"]

def calc_atr(df, length=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

@njit(fastmath=True)
def rolling_hurst_numba(values, window, max_lag=20):
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    lags = np.arange(2, max_lag)
    log_lags = np.log(lags)
    mean_x = np.mean(log_lags)
    dx = log_lags - mean_x
    ss_xx = np.sum(dx * dx)
    if ss_xx == 0:
        ss_xx = 1e-9
    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1: i + 1]
        tau = np.zeros(len(lags))
        for j in range(len(lags)):
            lag = lags[j]
            diff = window_data[lag:] - window_data[:-lag]
            m = np.mean(diff)
            v = np.mean((diff - m) ** 2)
            tau[j] = np.sqrt(v)
        log_tau = np.log(tau + 1e-9)
        mean_y = np.mean(log_tau)
        dy = log_tau - mean_y
        slope = np.sum(dx * dy) / ss_xx
        out[i] = slope * 2.0
    return out

@njit(fastmath=True)
def rolling_slope_numba(values, window):
    n = len(values)
    out = np.full(n, np.nan)
    x = np.arange(window).astype(np.float64)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        ss_xx = 1e-9
    for i in range(window - 1, n):
        y = values[i - window + 1: i + 1]
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        out[i] = ss_xy / ss_xx
    return out

@njit(fastmath=True)
def rolling_poc_numba(high, low, window, n_bins=20):
    n = len(high)
    poc_price = np.full(n, np.nan)
    poc_density = np.full(n, np.nan)
    for i in range(window - 1, n):
        h_seg = high[i - window + 1: i + 1]
        l_seg = low[i - window + 1: i + 1]
        seg_max = h_seg.max()
        seg_min = l_seg.min()
        if seg_max <= seg_min:
            continue
        bin_size = (seg_max - seg_min) / n_bins
        counts = np.zeros(n_bins)
        for j in range(window):
            start_bin = int((l_seg[j] - seg_min) / bin_size)
            end_bin = int((h_seg[j] - seg_min) / bin_size)
            if start_bin < 0: start_bin = 0
            if end_bin >= n_bins: end_bin = n_bins - 1
            for b in range(start_bin, end_bin + 1):
                counts[b] += 1.0
        max_bin = np.argmax(counts)
        poc_price[i] = seg_min + (max_bin + 0.5) * bin_size
        mean_count = counts.mean()
        poc_density[i] = counts[max_bin] / (mean_count + 1e-9)
    return poc_price, poc_density

def detect_causal_fractals(df, n=2):
    high, low = df['High'], df['Low']
    window = 2 * n + 1
    roll_max = high.rolling(window).max()
    roll_min = low.rolling(window).min()
    is_fractal_h = (high.shift(n) == roll_max) & high.shift(n).notna()
    is_fractal_l = (low.shift(n) == roll_min) & low.shift(n).notna()
    fractal_h_price = high.shift(n).where(is_fractal_h)
    fractal_l_price = low.shift(n).where(is_fractal_l)
    return is_fractal_h.fillna(False), is_fractal_l.fillna(False), fractal_h_price, fractal_l_price

def build_structure_features(df, n=2, prefix=""):
    idx_pos = pd.Series(np.arange(len(df)), index=df.index)
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
    out[f'{prefix}Swing_Range_ATR'] = (last_h_price - last_l_price).abs() / (atr + 1e-9)
    rng = (last_h_price - last_l_price).replace(0, np.nan)
    out[f'{prefix}Structure_Range_Position'] = ((df['Close'] - last_l_price) / rng).clip(0, 1)
    prev_h_price = last_h_price.where(is_fh).shift(1).ffill()
    high_class_event = np.where(is_fh & prev_h_price.notna(), np.where(fh_price > prev_h_price, 1, -1), 0)
    high_class = pd.Series(high_class_event, index=df.index).replace(0, np.nan).ffill().fillna(0)
    prev_l_price = last_l_price.where(is_fl).shift(1).ffill()
    low_class_event = np.where(is_fl & prev_l_price.notna(), np.where(fl_price > prev_l_price, 1, -1), 0)
    low_class = pd.Series(low_class_event, index=df.index).replace(0, np.nan).ffill().fillna(0)
    out[f'{prefix}Last_High_Class'] = high_class
    out[f'{prefix}Last_Low_Class'] = low_class
    structure_state = np.where((high_class == 1) & (low_class == 1), 1, np.where((high_class == -1) & (low_class == -1), -1, 0))
    out[f'{prefix}Market_Structure_State'] = structure_state
    state_series = pd.Series(structure_state, index=df.index)
    change_point = (state_series != state_series.shift(1)).cumsum()
    out[f'{prefix}Structure_Streak_Count'] = state_series.groupby(change_point).cumcount() + 1
    ref_high = last_h_price.shift(1)
    ref_low = last_l_price.shift(1)
    close_prev = df['Close'].shift(1)
    bos_bull = (close_prev <= ref_high) & (df['Close'] > ref_high)
    bos_bear = (close_prev >= ref_low) & (df['Close'] < ref_low)
    out[f'{prefix}BOS_Bull_Flag'] = bos_bull.astype(int)
    out[f'{prefix}BOS_Bear_Flag'] = bos_bear.astype(int)
    out[f'{prefix}BOS_Strength_ATR'] = np.where(bos_bull, (df['Close'] - ref_high) / (atr + 1e-9),
                                        np.where(bos_bear, (ref_low - df['Close']) / (atr + 1e-9), 0.0))
    bos_any_idx = idx_pos.where(bos_bull | bos_bear).ffill()
    out[f'{prefix}Bars_Since_BOS'] = (idx_pos - bos_any_idx)
    price_at_bos = df['Close'].where(bos_bull | bos_bear).ffill()
    out[f'{prefix}Trend_Leg_Magnitude_ATR'] = (df['Close'] - price_at_bos).abs() / (atr + 1e-9)
    prevailing_state = pd.Series(structure_state, index=df.index).shift(1).fillna(0)
    choch = ((bos_bull & (prevailing_state < 0)) | (bos_bear & (prevailing_state > 0))).astype(int)
    out[f'{prefix}CHoCH_Flag'] = choch
    choch_idx = idx_pos.where(choch.astype(bool)).ffill()
    out[f'{prefix}Bars_Since_CHoCH'] = (idx_pos - choch_idx)
    half_life = 10.0
    bos_strength_at_event = pd.Series(np.where(bos_bull | bos_bear, out[f'{prefix}BOS_Strength_ATR'], np.nan), index=df.index).ffill().fillna(0)
    out[f'{prefix}BOS_Impulse_Decay'] = bos_strength_at_event * np.exp(-out[f'{prefix}Bars_Since_BOS'].fillna(9999) / half_life)
    out[f'{prefix}CHoCH_Recency_Decay'] = np.exp(-out[f'{prefix}Bars_Since_CHoCH'].fillna(9999) / half_life)
    bos_dir_event = np.where(bos_bull, 1, np.where(bos_bear, -1, np.nan))
    bos_dir = pd.Series(bos_dir_event, index=df.index).ffill().fillna(0)
    bos_dir_change = (bos_dir != bos_dir.shift(1)).cumsum()
    out[f'{prefix}BOS_Same_Direction_Streak'] = bos_dir.groupby(bos_dir_change).cumcount() + 1
    return out.shift(1)  # 🛡️ FIX: کل بلوک یک کندل عقب می‌افتد

def build_trend_context_features(df):
    out = pd.DataFrame(index=df.index)
    atr14 = calc_atr(df, 14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    out['RSI_14'] = (100 - (100 / (1 + rs))).shift(1)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out['MACD_Hist'] = (macd - macd_signal).shift(1)
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    out['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean().shift(1)
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    ma21 = df['Close'].rolling(21).mean()
    out['Price_to_EMA200_ATR'] = ((df['Close'] - ema200) / (atr14 + 1e-9)).shift(1)
    out['Price_to_MA21_ATR'] = ((df['Close'] - ma21) / (atr14 + 1e-9)).shift(1)
    tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    out['Tenkan_Kijun_Dist_ATR'] = ((tenkan - kijun) / (atr14 + 1e-9)).shift(1)
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2)
    kumo_mid = ((senkou_a + senkou_b) / 2).shift(26)
    out['Price_to_Kumo_ATR'] = ((df['Close'] - kumo_mid) / (atr14 + 1e-9)).shift(1)
    return out

def build_fractal_regime_features(df, window=100):
    out = pd.DataFrame(index=df.index)
    log_price = np.log(df['Close'].replace(0, np.nan)).ffill().values
    hurst = rolling_hurst_numba(log_price, window=window, max_lag=20)
    out['Hurst_Exponent'] = pd.Series(hurst, index=df.index).shift(1)
    out['Fractal_Dimension'] = (2.0 - out['Hurst_Exponent'])
    safe_high, safe_low = df['High'], df['Low']
    out['Fractal_Range_Ratio_10_20'] = ((safe_high.rolling(10).max() - safe_low.rolling(10).min()) / ((safe_high.rolling(20).max() - safe_low.rolling(20).min()) + 1e-9)).shift(1)
    return out

def build_longterm_trend_features(df):
    out = pd.DataFrame(index=df.index)
    atr14 = calc_atr(df, 14)
    close_vals = df['Close'].values.astype(np.float64)
    for w in (50, 100, 200):
        slope = rolling_slope_numba(close_vals, w)
        out[f'Slope_{w}_ATR'] = (pd.Series(slope, index=df.index) / (atr14 + 1e-9)).shift(1)
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    ema100 = df['Close'].ewm(span=100, adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    stack = (np.where(ema50 > ema100, 1, -1) + np.where(ema100 > ema200, 1, -1) + np.where(ema50 > ema200, 1, -1))
    out['EMA_Stack_Score'] = pd.Series(stack, index=df.index).shift(1)
    return out

def build_fibonacci_features(df, n=2):
    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=n)
    last_h = fh_price.ffill()
    last_l = fl_price.ffill()
    atr14 = calc_atr(df, 14)
    rng = (last_h - last_l).replace(0, np.nan)
    pos = ((df['Close'] - last_l) / rng)
    fib_levels = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
    pos_vals = pos.values
    valid = ~np.isnan(pos_vals)
    nearest_level = np.full(len(pos_vals), np.nan)
    if valid.any():
        diffs = np.abs(pos_vals[valid][:, None] - fib_levels[None, :])
        nearest_level[valid] = fib_levels[diffs.argmin(axis=1)]
    dist_ratio = pos_vals - nearest_level
    dist_atr = dist_ratio * rng.values / (atr14.values + 1e-9)
    out = pd.DataFrame(index=df.index)
    out['Fib_Position_In_Leg'] = pos.shift(1)
    out['Nearest_Fib_Ratio'] = pd.Series(nearest_level, index=df.index).shift(1)
    out['Dist_To_Nearest_Fib_ATR'] = pd.Series(dist_atr, index=df.index).shift(1)
    return out

def build_sr_strength_features(df, n=2, lookback=100, touch_thresh_atr=0.25):
    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=n)
    last_h = fh_price.ffill()
    last_l = fl_price.ffill()
    atr14 = calc_atr(df, 14)
    thresh = touch_thresh_atr * atr14
    near_h = (df['High'] >= (last_h - thresh)) & (df['High'] <= (last_h + thresh))
    near_l = (df['Low'] <= (last_l + thresh)) & (df['Low'] >= (last_l - thresh))
    out = pd.DataFrame(index=df.index)
    out['SwingHigh_Touch_Count'] = near_h.rolling(lookback).sum().shift(1)
    out['SwingLow_Touch_Count'] = near_l.rolling(lookback).sum().shift(1)
    return out

def build_order_block_features(df, disp_mult=1.5):
    atr14 = calc_atr(df, 14)
    body = df['Close'] - df['Open']
    is_bear_candle = body < 0
    is_bull_candle = body > 0
    is_displacement = body.abs() >= (disp_mult * atr14)
    bull_ob_event = is_displacement & (body > 0) & is_bear_candle.shift(1).fillna(False)
    bear_ob_event = is_displacement & (body < 0) & is_bull_candle.shift(1).fillna(False)
    ob_bull_mid = ((df['High'].shift(1) + df['Low'].shift(1)) / 2).where(bull_ob_event).ffill()
    ob_bear_mid = ((df['High'].shift(1) + df['Low'].shift(1)) / 2).where(bear_ob_event).ffill()
    idx_pos = pd.Series(np.arange(len(df)), index=df.index)
    bull_ob_idx = idx_pos.where(bull_ob_event).ffill()
    bear_ob_idx = idx_pos.where(bear_ob_event).ffill()
    out = pd.DataFrame(index=df.index)
    out['Dist_To_Bull_OB_ATR'] = ((df['Close'] - ob_bull_mid) / (atr14 + 1e-9)).shift(1)
    out['Dist_To_Bear_OB_ATR'] = ((df['Close'] - ob_bear_mid) / (atr14 + 1e-9)).shift(1)
    out['Bars_Since_Bull_OB'] = (idx_pos - bull_ob_idx).shift(1)
    out['Bars_Since_Bear_OB'] = (idx_pos - bear_ob_idx).shift(1)
    return out

def build_volume_free_profile_features(df, window=100, n_bins=20):
    high = np.ascontiguousarray(df['High'].values, dtype=np.float64)
    low = np.ascontiguousarray(df['Low'].values, dtype=np.float64)
    poc_price, poc_density = rolling_poc_numba(high, low, window, n_bins)
    atr14 = calc_atr(df, 14)
    out = pd.DataFrame(index=df.index)
    out['Dist_To_POC_ATR'] = ((df['Close'] - pd.Series(poc_price, index=df.index)) / (atr14 + 1e-9)).shift(1)
    out['POC_Density_Score'] = pd.Series(poc_density, index=df.index).shift(1)
    return out

def build_pivot_wave_features(df, n=2):
    out = pd.DataFrame(index=df.index)
    idx_pos = pd.Series(np.arange(len(df)), index=df.index)
    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=n)
    atr = calc_atr(df, 14)
    last_h_price = fh_price.ffill()
    last_l_price = fl_price.ffill()
    last_h_idx = idx_pos.where(is_fh).ffill()
    last_l_idx = idx_pos.where(is_fl).ffill()
    last_is_high = (last_h_idx.fillna(-1) > last_l_idx.fillna(-1))
    last_pivot_price = pd.Series(np.where(last_is_high, last_h_price, last_l_price), index=df.index)
    last_pivot_idx = pd.Series(np.where(last_is_high, last_h_idx, last_l_idx), index=df.index)
    opp_last_idx = pd.Series(np.where(last_is_high, last_l_idx, last_h_idx), index=df.index)
    opp_last_price = pd.Series(np.where(last_is_high, last_l_price, last_h_price), index=df.index)
    ab_range = (last_pivot_price - opp_last_price).abs()
    ab_bars = (last_pivot_idx - opp_last_idx)
    out['PW_AB_Range_ATR'] = ab_range / (atr + 1e-9)
    out['PW_AB_Bars'] = ab_bars
    out['PW_Bars_Since_Pivot'] = (idx_pos - last_pivot_idx)
    out['PW_BC_AB_Bar_Ratio'] = out['PW_Bars_Since_Pivot'] / ab_bars.replace(0, np.nan)
    sign = np.where(last_is_high, 1.0, -1.0)
    out['PW_Retrace_Ratio'] = ((last_pivot_price - df['Close']) / (ab_range.replace(0, np.nan))) * sign
    change_point = (last_pivot_idx != last_pivot_idx.shift(1)).cumsum()
    seg_high_since_pivot = df['High'].groupby(change_point).cummax()
    seg_low_since_pivot = df['Low'].groupby(change_point).cummin()
    out['PW_Wick_Beyond_Origin_Flag'] = np.where(last_is_high, (seg_low_since_pivot < opp_last_price).astype(int), (seg_high_since_pivot > opp_last_price).astype(int))
    return out.shift(1)

def build_volatility_features(df):
    out = pd.DataFrame(index=df.index)
    atr14 = calc_atr(df, 14)
    atr100 = calc_atr(df, 100)
    out['ATR_Normalized'] = (atr14 / df['Close']).shift(1)
    out['Volatility_Regime_Ratio'] = (atr14 / (atr100 + 1e-9)).shift(1)
    log_hl = (np.log(df['High'] / df['Low'])) ** 2
    log_co = (np.log(df['Close'] / df['Open'])) ** 2
    gk_vol = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
    out['Garman_Klass_Volatility_20'] = gk_vol.rolling(20).mean().shift(1)
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    out['BB_Width_ATR'] = ((2 * std20) / (atr14 + 1e-9)).shift(1)
    out['Squeeze_State'] = (out['BB_Width_ATR'] < out['BB_Width_ATR'].rolling(50).quantile(0.2)).astype(int).shift(1)
    bbw_mean50 = out['BB_Width_ATR'].rolling(50).mean()
    bbw_std50 = out['BB_Width_ATR'].rolling(50).std()
    out['Squeeze_Zscore'] = (out['BB_Width_ATR'] - bbw_mean50) / (bbw_std50 + 1e-9)
    out['Squeeze_Zscore'] = out['Squeeze_Zscore'].shift(1)
    vol5 = df['Close'].pct_change().rolling(5).std()
    vol40 = df['Close'].pct_change().rolling(40).std()
    out['MultiScale_Volatility_Ratio_5_40'] = (vol5 / (vol40 + 1e-9)).shift(1)
    z = (df['Close'] - ma20) / (std20 + 1e-9)
    out['Mean_Reversion_ZScore_20'] = z.shift(1)
    return out

def build_liquidity_features(df):
    # 🛡️ FIXED: قبلا بدون shift بود → نشت
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
    return out.shift(1)  # FIX

def build_time_features(df):
    out = pd.DataFrame(index=df.index)
    out['Time_Sine'] = np.sin(2 * np.pi * df.index.hour / 24)
    out['Time_Cosine'] = np.cos(2 * np.pi * df.index.hour / 24)
    hour = df.index.hour + df.index.minute/60.0
    out['London_Killzone'] = ((hour >= 8) & (hour <= 11)).astype(int)
    out['NY_Killzone'] = ((hour >= 13) & (hour <= 16)).astype(int)
    out['Asia_Killzone'] = ((hour >= 0) & (hour <= 4)).astype(int)
    out['Is_London_NY_Overlap'] = ((hour >= 13) & (hour <= 16)).astype(int)
    return out.shift(1)

HTF_TIMEFRAME_MAP = {
    'M1': ['15min', '1h'], 'M5': ['1h', '4h'], 'M15': ['1h', '4h'],
    'M30': ['4h', '1D'], 'H1': ['4h', '1D'], 'H4': ['1D', '1W'], 'D1': ['1W', '1ME'],
}

def build_mtf_structure_features(df, resample_rule, tag):
    df_htf = df[['Open', 'High', 'Low', 'Close']].resample(resample_rule).agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    ).dropna()
    if len(df_htf) < 30:
        return pd.DataFrame(index=df.index)
    htf_struct = build_structure_features(df_htf, n=2, prefix=f"MTF_{tag}_")
    htf_struct_shifted = htf_struct.shift(1)
    combined_index = df.index.union(htf_struct_shifted.index).sort_values()
    merged = htf_struct_shifted.reindex(combined_index).ffill().reindex(df.index)
    return merged.shift(1)  # extra safety

def build_mtf_alignment_features(micro_struct, htf_struct_list, tags):
    out = pd.DataFrame(index=micro_struct.index)
    states = [micro_struct['Market_Structure_State']]
    for htf_struct, tag in zip(htf_struct_list, tags):
        col = f'MTF_{tag}_Market_Structure_State'
        if col in htf_struct.columns:
            states.append(htf_struct[col])
    if len(states) > 1:
        stacked = pd.concat(states, axis=1).fillna(0)
        out['MTF_Alignment_Score'] = stacked.sum(axis=1) / len(states)
    else:
        out['MTF_Alignment_Score'] = 0.0
    return out.shift(1)

def process_data_batch(pairs, tf, micro_n, macro_n, feature_groups, progress=gr.Progress()):
    if not pairs or "No Tick Data Found" in pairs:
        yield "❌ خطا: هیچ فایلی برای پردازش انتخاب نشده.", pd.DataFrame()
        return
    tf_map = {'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min', 'H1': '1h', 'H4': '4h', 'D1': '1D'}
    resample_rule = tf_map.get(tf, '15min')
    total_pairs = len(pairs)
    final_message = f"🚀 عملیات روی {total_pairs} نماد آغاز شد (Structure Engine v24 FIXED - Zero Leak)...\n"
    last_tail_df = pd.DataFrame()
    yield final_message, last_tail_df

    for pair_idx, pair in enumerate(pairs):
        chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, f"{pair}_Tick*.parquet")))
        if not chunk_files:
            final_message += f"⚠️ پرش از {pair}: فایل یافت نشد.\n"
            yield final_message, last_tail_df
            continue
        base_prog = pair_idx / total_pairs
        prog_step = 1.0 / total_pairs
        try:
            resampled_chunks = []
            for c_idx, filepath in enumerate(chunk_files):
                progress(base_prog + (0.10 * prog_step * (c_idx / len(chunk_files))), desc=f"[{pair}] ⏳ Loading Tick Chunk {c_idx+1}/{len(chunk_files)}...")
                df_raw = pd.read_parquet(filepath, columns=['Bid'])
                if not isinstance(df_raw.index, pd.DatetimeIndex):
                    df_raw.index = pd.to_datetime(df_raw.index)
                tick_diff = df_raw['Bid'].diff().fillna(0)
                df_raw['Tick_Up'] = (tick_diff > 0).astype(np.int8)
                df_raw['Tick_Down'] = (tick_diff < 0).astype(np.int8)
                df_chunk = df_raw.resample(resample_rule).agg(
                    Open=('Bid', 'first'), High=('Bid', 'max'), Low=('Bid', 'min'), Close=('Bid', 'last'),
                    Volume=('Bid', 'count'), Tick_Up_Count=('Tick_Up', 'sum'), Tick_Down_Count=('Tick_Down', 'sum')
                ).dropna()
                resampled_chunks.append(df_chunk)
                del df_raw, tick_diff
                gc.collect()
            if not resampled_chunks:
                continue
            progress(base_prog + (0.30 * prog_step), desc=f"[{pair}] 🔗 Merging All Chunks...")
            df = pd.concat(resampled_chunks).sort_index()
            df = df[~df.index.duplicated(keep='first')]
            del resampled_chunks
            gc.collect()
        except Exception:
            final_message += f"❌ خطای بحرانی در {pair}:\n{traceback.format_exc()}\n"
            yield final_message, last_tail_df
            continue

        progress(base_prog + (0.40 * prog_step), desc=f"[{pair}] 🧠 Structure Narrative (Micro)...")
        struct_micro = build_structure_features(df, n=int(micro_n), prefix="")

        progress(base_prog + (0.55 * prog_step), desc=f"[{pair}] 🧭 Structure Narrative (Macro)...")
        struct_macro = build_structure_features(df, n=int(macro_n), prefix="HTF_")

        progress(base_prog + (0.65 * prog_step), desc=f"[{pair}] 📊 Volatility & Liquidity...")
        vol_feats = build_volatility_features(df)
        liq_feats = build_liquidity_features(df)
        time_feats = build_time_features(df)

        progress(base_prog + (0.75 * prog_step), desc=f"[{pair}] 📈 Trend Context & Fractal Regime...")
        trend_feats = build_trend_context_features(df) if "Trend Context (RSI/ADX/MACD/Ichimoku)" in feature_groups else pd.DataFrame(index=df.index)
        fractal_feats = build_fractal_regime_features(df) if "Fractal Regime (Hurst/Fractal Dimension)" in feature_groups else pd.DataFrame(index=df.index)

        progress(base_prog + (0.82 * prog_step), desc=f"[{pair}] 🧭 Long-Term Trend / Fibonacci...")
        longterm_feats = build_longterm_trend_features(df) if "Long-Term Trend Slope (50/100/200)" in feature_groups else pd.DataFrame(index=df.index)
        fib_feats = build_fibonacci_features(df, n=int(micro_n)) if "Fibonacci Retracement (Swing-Relative)" in feature_groups else pd.DataFrame(index=df.index)
        pivot_wave_feats = build_pivot_wave_features(df, n=int(micro_n)) if "Pivot Wave Context (AB/BC Awareness)" in feature_groups else pd.DataFrame(index=df.index)

        progress(base_prog + (0.87 * prog_step), desc=f"[{pair}] 🧱 S/R Strength & Order Blocks...")
        sr_feats = build_sr_strength_features(df, n=int(micro_n)) if "Support/Resistance Strength" in feature_groups else pd.DataFrame(index=df.index)
        ob_feats = build_order_block_features(df) if "Order Blocks (ICT)" in feature_groups else pd.DataFrame(index=df.index)

        progress(base_prog + (0.90 * prog_step), desc=f"[{pair}] 📦 Volume-Free Price Profile...")
        profile_feats = build_volume_free_profile_features(df) if "Price Profile (Volume-Free POC)" in feature_groups else pd.DataFrame(index=df.index)

        mtf_blocks = []
        if "Higher-Timeframe Structure (Real MTF, Anti-Leak)" in feature_groups:
            progress(base_prog + (0.93 * prog_step), desc=f"[{pair}] 🕰️ Real Multi-Timeframe Structure...")
            htf_rules = HTF_TIMEFRAME_MAP.get(tf, ['4h', '1D'])
            htf_tags = ['HTF1', 'HTF2']
            htf_struct_frames = []
            for rule, tag in zip(htf_rules, htf_tags):
                htf_feat = build_mtf_structure_features(df, rule, tag)
                mtf_blocks.append(htf_feat)
                htf_struct_frames.append(htf_feat)
            align_feat = build_mtf_alignment_features(struct_micro, htf_struct_frames, htf_tags)
            mtf_blocks.append(align_feat)

        all_blocks = [df]
        if "Market Structure (Micro)" in feature_groups: all_blocks.append(struct_micro)
        if "Market Structure (Macro / HTF)" in feature_groups: all_blocks.append(struct_macro)
        if "Volatility & Regime" in feature_groups: all_blocks.append(vol_feats)
        if "Liquidity & Tick Pressure" in feature_groups: all_blocks.append(liq_feats)
        if "Time Context" in feature_groups: all_blocks.append(time_feats)
        all_blocks.extend([trend_feats, fractal_feats, longterm_feats, fib_feats, sr_feats, ob_feats, profile_feats, pivot_wave_feats])
        all_blocks.extend(mtf_blocks)

        df_final = pd.concat(all_blocks, axis=1)
        df_final = df_final.replace([np.inf, -np.inf], np.nan)

        # 🛡️ FIX: حذف Warmup + NaN به جای fillna(0)
        warmup = 250
        rows_before = len(df_final)
        df_final = df_final.iloc[warmup:]
        # گزارش NaN قبل از drop
        nan_frac = df_final.isna().mean().sort_values(ascending=False)
        bad_cols = nan_frac[nan_frac > 0.3]
        if len(bad_cols) > 0:
            final_message += f"⚠️ [{pair}] {len(bad_cols)} فیچر با >30% NaN حذف شد: {list(bad_cols.index[:5])}\n"
            df_final = df_final.drop(columns=bad_cols.index)
        df_final = df_final.dropna()
        rows_after = len(df_final)
        final_message += f"   Warmup+Drop: {rows_before} → {rows_after} ردیف (حذف {rows_before-rows_after})\n"

        progress(base_prog + (0.90 * prog_step), desc=f"[{pair}] 💾 Saving Parquet...")
        out_path = os.path.join(DATA_DIR, f"{pair}_Features.parquet")
        df_final.to_parquet(out_path, compression='snappy')

        final_message += f"✅ {pair} تکمیل شد | ردیف‌ها: {len(df_final)} | ستون‌ها: {len(df_final.columns)}\n"
        last_tail_df = df_final.tail(50).reset_index()
        yield final_message, last_tail_df
        del df, df_final, struct_micro, struct_macro, vol_feats, liq_feats, time_feats, trend_feats, fractal_feats, longterm_feats, fib_feats, sr_feats, ob_feats, profile_feats, pivot_wave_feats, mtf_blocks
        gc.collect()

    final_message += "\n🏁 پایان عملیات Structure Engine FIXED."
    yield final_message, last_tail_df

with gr.Blocks(title="HIPO STRUCTURE ENGINE v24 FIXED") as web_app:
    gr.HTML("""
        <div style="text-align: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 20px;">
            <h1 style="color: #00ff88; font-family: monospace; font-size: 30px;">🧠 HIPO STRUCTURE ENGINE v24 FIXED</h1>
            <p style="color: #00f2ff; font-family: monospace;">Zero Leak (all shift1) + Killzone + Warmup Drop 250</p>
        </div>
    """)
    with gr.Row():
        w_pairs = gr.CheckboxGroup(choices=get_available_pairs(), label="1️⃣ انتخاب نمادها", value=[get_available_pairs()[0]] if get_available_pairs() else [])
        w_tf = gr.Dropdown(choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'], value='M15', label="2️⃣ تایم‌فریم پایه")
    with gr.Row():
        w_micro_n = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="حساسیت فرکتال خرد")
        w_macro_n = gr.Slider(minimum=5, maximum=30, step=1, value=8, label="حساسیت فرکتال کلان")
    FEATURE_GROUP_CHOICES = [
        "Market Structure (Micro)", "Market Structure (Macro / HTF)",
        "Volatility & Regime", "Liquidity & Tick Pressure", "Time Context",
        "Trend Context (RSI/ADX/MACD/Ichimoku)", "Fractal Regime (Hurst/Fractal Dimension)",
        "Long-Term Trend Slope (50/100/200)", "Fibonacci Retracement (Swing-Relative)",
        "Support/Resistance Strength", "Order Blocks (ICT)", "Price Profile (Volume-Free POC)",
        "Higher-Timeframe Structure (Real MTF, Anti-Leak)",
        "Pivot Wave Context (AB/BC Awareness)"
    ]
    w_groups = gr.CheckboxGroup(choices=FEATURE_GROUP_CHOICES, value=FEATURE_GROUP_CHOICES, label="گروه‌های فیچر فعال")
    w_btn = gr.Button("🚀 EXTRACT STRUCTURE FEATURES (FIXED)", variant="primary", size="lg")
    w_msg = gr.Textbox(label="📡 وضعیت عملیات", lines=12)
    w_tail = gr.DataFrame(label="📊 نمونه‌ی ۵۰ ردیف آخر خروجی")
    w_btn.click(process_data_batch, inputs=[w_pairs, w_tf, w_micro_n, w_macro_n, w_groups], outputs=[w_msg, w_tail])

web_app.queue().launch(share=True, inbrowser=True)
