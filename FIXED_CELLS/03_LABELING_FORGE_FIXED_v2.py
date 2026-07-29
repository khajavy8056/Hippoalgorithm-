# @title 🧭 HIPO LABELING FORGE [v5.0 COMPLETE — GA Winners>Losers + Feature Quality Report] { display-mode: "form" }
# =============================================================================
# سلول کامل لیبل‌گذاری Pivot Settlement — آماده Colab
# ورودی : /content/hipo_lab_data/*_Features.parquet  (+ Tick files)
# خروجی : /content/hipo_lab_data/*_Labeled.parquet + گزارش کیفیت فیچر
#
# تب‌ها:
#  1) استخراج دستی
#  2) اپتیمایز ژنتیک چندهدفه
#       اهداف سخت/نرم:
#         • Wins_Class1 > Losses_Class0  (قید اصلی: بردها بیشتر از باخت‌ها)
#         • تعداد نمونه کافی (min_samples)
#         • Feature_Quality_Score بالا (هم‌راستایی فیچر↔لیبل)
#  3) گزارش کیفیت فیچر↔لیبل (بعد از لیبل)
#
# هماهنگ با Feature Engine v25 (PSA/LSW/FVG)
# =============================================================================


# =============================================================================
# ستاپ «پیوت تسویه» (Pivot Settlement Setup) — بر اساس PDF جلیل ضرغام
# =============================================================================
# خلاصه‌ی منطق (کاملاً علّی، بدون هیچ نگاه به آینده):
#
#   A --(موج ایمپالسیو AB، شارپ و قدرتمند)--> B
#   B --(اصلاح BC، بین ۲۰ تا ۵۰٪ AB، حداقل ۳ کندل)--> C
#   [باکس نقدینگی به‌اندازه‌ی AB، منتقل‌شده به نقطه‌ی C]
#   C --(تلاش برای شکست سطح B — احتمالاً یک شکار نقدینگی/تله)--> کندل سیگنال
#   کندل سیگنال --(n کندلِ میانی مجاز، در چارچوب باکس)--> کندل تایید/تریگر
#   کندل تایید = شکست بدنه‌ایِ لوی کندل سیگنال (برای ستاپ فروش) → ورود معامله
#
# این یک ستاپ *ریورسال* در نقطه‌ی پیوت است: وقتی AB صعودی باشد (A=کف, B=سقف)،
# کندل سیگنال یک شکار نقدینگیِ صعودی فراتر از B است که واقعاً یک تله برای
# خریداران است؛ کندل تایید با شکستِ لوی کندل سیگنال، ریورسال نزولی (SELL) را
# تأیید می‌کند. حالت قرینه (AB نزولی، A=سقف, B=کف) یک ستاپ BUY می‌سازد.
#
# 🛡️ قوانین ضد نشت دیتا (Zero Lookahead) که در این موتور رعایت شده‌اند:
#   ۱. پیوت‌های A و B فقط با فرکتال n-کندلیِ *تاییدشده* شناسایی می‌شوند (دقیقاً
#      همان موتور علّی build_structure_features/detect_causal_fractals از سلول
#      فیچرها) — یعنی قیمتِ پیوت B فقط از لحظه‌ی confirm_idx = pivot_idx + n
#      به بعد "شناخته‌شده" فرض می‌شود، نه زودتر.
#   ۲. جست‌وجوی BC/باکس/کندل سیگنال/کندل تایید همیشه از یک اندیس > confirm_idx
#      شروع می‌شود و در هر قدم فقط از کندل‌های قبلاً بسته‌شده استفاده می‌کند.
#   ۳. ورود معامله همیشه در Open کندلِ *بلافاصله پس از* کندل تایید رخ می‌دهد
#      (نه در همان کندل تایید) — دقیقاً هم‌سو با قرارداد بقیه‌ی موتورهای پروژه
#      (Channel Breakout v20 / ICT Silver Bullet v21).
#   ۴. داوری نتیجه (برد/باخت) با موتور مسابقه‌ی تیک (Tick Racing) روی دیتای
#      خام تیک انجام می‌شود، دقیقاً مثل بقیه‌ی سلول‌های لیبل‌گذاری پروژه.
# =============================================================================

import sys, subprocess

def smart_install():
    reqs = ["gradio", "pyarrow", "matplotlib", "numba", "pandas", "numpy"]
    missing = [req for req in reqs if req not in sys.modules]
    if missing:
        print(f"⏳ Installing Infrastructure ({', '.join(missing)})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        from IPython.display import clear_output
        clear_output()

smart_install()

import os, glob, gc, shutil, json
from collections import Counter
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from numba import njit

# =============================================================================
# 2. تنظیمات پایه و مسیرها (Paths) — دقیقاً هم‌سو با بقیه‌ی سلول‌های پروژه
# =============================================================================
DATA_DIR = "/content/hipo_lab_data"
IMG_DIR = "/content/signals_img_pivot_settlement"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def get_available_datasets():
    files = glob.glob(os.path.join(DATA_DIR, "*_Features.parquet"))
    return list(set([os.path.basename(f).replace("_Features.parquet", "") for f in files])) if files else ["No Data Found"]

# =============================================================================
# 3. مدیر هوشمند دیتای تیک (Smart Tick Chunk Locator) — بدون تغییر نسبت به بقیه‌ی سلول‌ها
# =============================================================================

class TickManager:
    """نسخه FIXED: از cross-year پشتیبانی می‌کند و دو فایل را concat می‌کند. هماهنگ با فیوچر v24."""
    def __init__(self, data_dir, pair):
        self.data_dir = data_dir
        self.pair = pair.upper()
        self.chunks = []
        self.current_files = {}
        self._map_chunks()

    def _map_chunks(self):
        files = glob.glob(os.path.join(self.data_dir, f"{self.pair}_Tick_*.parquet"))
        for f in files:
            parts = os.path.basename(f).replace(".parquet", "").split("_")
            if len(parts) >= 4:
                try:
                    self.chunks.append((int(parts[-2]), int(parts[-1]), f))
                except:
                    pass

    def get_ticks_series(self, start_time, end_time):
        needed_files = []
        for sy, ey, fpath in self.chunks:
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
                sliced = df.loc[start_time:end_time]['Bid']
                if len(sliced)>0:
                    dfs.append(sliced)
            except:
                continue
        if not dfs:
            return pd.Series([], dtype=np.float64)
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined

    def clear_memory(self):
        self.current_files.clear()
        gc.collect()

def calc_raw_atr(df, length=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


@njit(fastmath=True)
def simulate_race_ps(ticks, dir_signal, sl_price, tp_price, spread_raw):
    """موتور مسابقه‌ی تیک — همان قرارداد داوری بقیه‌ی سلول‌های پروژه."""
    for i in range(len(ticks)):
        price = ticks[i]
        if dir_signal == 1:  # Buy
            if price <= sl_price: return 0, i
            elif price >= tp_price: return 1, i
        else:  # Sell
            if price >= (sl_price - spread_raw): return 0, i
            elif price <= (tp_price - spread_raw): return 1, i
    return -1, len(ticks) - 1

def estimate_real_spread(pair):
    p = pair.upper()
    if "XAU" in p or "GOLD" in p:
        return 0.35
    elif "JPY" in p:
        return 0.015
    else:
        return 0.00012



# =============================================================================
# 5. موتور فرکتال علّی (همان منطق دقیق سلول فیچرها — یک‌بار دیگر اینجا تعریف
#    شده تا این سلول کاملاً مستقل و خودکفا باشد و وابسته به اجرای سلول فیچرها
#    در همین Runtime نباشد)
# =============================================================================
def detect_causal_fractals(df, n=2):
    """
    فرکتال سقف/کف با تاییدِ n کندل سمت راست (کاملاً علّی).
    خروجی: بولین + قیمتِ فرکتالِ تازه‌تایید‌شده، دقیقاً در ردیفی که تایید می‌شود.
    """
    high, low = df['High'], df['Low']
    window = 2 * n + 1
    roll_max = high.rolling(window).max()
    roll_min = low.rolling(window).min()

    is_fractal_h = (high.shift(n) == roll_max) & high.shift(n).notna()
    is_fractal_l = (low.shift(n) == roll_min) & low.shift(n).notna()

    fractal_h_price = high.shift(n).where(is_fractal_h)
    fractal_l_price = low.shift(n).where(is_fractal_l)

    return is_fractal_h.fillna(False), is_fractal_l.fillna(False), fractal_h_price, fractal_l_price


def build_causal_zigzag(is_fh, is_fl, fh_price, fl_price, swing_n):
    """
    از خروجی فرکتال علّی، یک دنباله‌ی زیگزاگِ متناوب (پیوت‌های سقف/کف پشت‌سرهم،
    هرکدوم مخالف قبلی) می‌سازد. این کار کاملاً علّی است چون فقط با ترتیب زمانیِ
    confirm_idx (نه pivot_idx) پیمایش می‌کنیم — یعنی هر پیوت دقیقاً همان لحظه‌ای
    که در دنیای واقعی «رسماً شناخته می‌شود» وارد دنباله می‌شود.
    هر آیتم: dict(pivot_idx, confirm_idx, price, type) با type در {'H','L'}
    """
    n = len(is_fh)
    events = []
    fh_vals = fh_price.values
    fl_vals = fl_price.values
    fh_flags = is_fh.values
    fl_flags = is_fl.values

    for i in range(n):
        if fh_flags[i]:
            events.append((i, i - swing_n, fh_vals[i], 'H'))
        if fl_flags[i]:
            events.append((i, i - swing_n, fl_vals[i], 'L'))
    events.sort(key=lambda e: e[0])

    zigzag = []
    for confirm_idx, pivot_idx, price, typ in events:
        if zigzag and zigzag[-1]['type'] == typ:
            if (typ == 'H' and price > zigzag[-1]['price']) or (typ == 'L' and price < zigzag[-1]['price']):
                zigzag[-1] = {'pivot_idx': pivot_idx, 'confirm_idx': confirm_idx, 'price': price, 'type': typ}
        else:
            zigzag.append({'pivot_idx': pivot_idx, 'confirm_idx': confirm_idx, 'price': price, 'type': typ})
    return zigzag

# =============================================================================
# 6. منطق ساختاری اصلی ستاپ «پیوت تسویه» (کاملاً علّی)
# =============================================================================
def _candle_quality(open_, high, low, close, atr_val, body_ratio_min, wick_reject_ratio, atr_mult_min, is_bull_break):
    """
    بررسیِ کیفیتِ یک کندل سیگنال منفرد طبق قوانین PDF:
      - ATR-تایید (رِنج کندل نسبت به ATR)
      - بدنه‌ی معقول (نه یک دوجی بی‌معنی)
      - شدوی مخالفِ جهتِ شکست نباید زیادی بزرگ باشد (نشانه‌ی فشار مخالف/تله‌ی
        زودهنگام که باعث می‌شه نتونیم بهش اعتماد کنیم)
    is_bull_break=True یعنی این کندل به‌سمت بالا سطح B رو شکسته (سناریوی فروش،
    AB صعودی)؛ False یعنی به‌سمت پایین شکسته (سناریوی خرید، AB نزولی).
    """
    rng = high - low
    if rng <= 0 or atr_val <= 0 or np.isnan(atr_val):
        return False
    body = abs(close - open_)
    if (rng / atr_val) < atr_mult_min:
        return False
    if (body / rng) < body_ratio_min:
        return False
    if is_bull_break:
        opposite_wick = high - max(open_, close)  # شدوی بالا (مخالفِ ادامه‌ی نزولیِ بعدی که انتظار داریم)
    else:
        opposite_wick = min(open_, close) - low    # شدوی پایین
    if body > 0 and (opposite_wick / body) > wick_reject_ratio:
        return False
    return True


def build_pivot_settlement_signals(df, swing_n, ab_min_bars, ab_max_bars, ab_atr_mult_min,
                                    ab_extended_min, ab_extended_max, bc_min_bars,
                                    bc_retrace_min, bc_retrace_max, box_scale,
                                    signal_search_bars, signal_atr_mult, signal_body_ratio_min,
                                    signal_wick_reject_ratio, allow_fl_candle,
                                    confirm_search_bars, confirm_atr_mult, sl_buffer_atr_mult,
                                    cd_ab_mode="CD > AB (طبق نقاط تکرارشده در PDF)",
                                    ab_noise_fraction_max=0.34, ab2_discount_mult=0.8):
    """
    موتور اصلی کشف ستاپ «پیوت تسویه» — یک اسکن رویدادمحور روی دنباله‌ی زیگزاگ
    پیوت‌های علّی (نه اسکن کندل‌به‌کندلِ کل دیتافریم)، دقیقاً هم‌سو با سبک
    build_silver_bullet_signals در سلول ICT Silver Bullet پروژه.
    """
    n_bars = len(df)
    is_fh, is_fl, fh_price, fl_price = detect_causal_fractals(df, n=swing_n)
    zigzag = build_causal_zigzag(is_fh, is_fl, fh_price, fl_price, swing_n)

    atr = calc_raw_atr(df, 14).values
    open_arr = df['Open'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    close_arr = df['Close'].values

    signals = []
    funnel = Counter()
    funnel['00_Total_AB_Candidates (zigzag pairs)'] = max(0, len(zigzag) - 1)

    for k in range(len(zigzag) - 1):
        A, B = zigzag[k], zigzag[k + 1]
        pivot_idx_A, pivot_idx_B = A['pivot_idx'], B['pivot_idx']
        if pivot_idx_A < 0 or pivot_idx_B < 0 or pivot_idx_B <= pivot_idx_A:
            funnel['01_Reject_Invalid_Pivot_Index'] += 1
            continue

        bars_AB = pivot_idx_B - pivot_idx_A
        ab_range = abs(B['price'] - A['price'])
        atr_at_B = atr[pivot_idx_B]
        if np.isnan(atr_at_B) or atr_at_B <= 0 or ab_range <= 0:
            funnel['02_Reject_Invalid_ATR_or_Range'] += 1
            continue

        ab_atr_ratio = ab_range / atr_at_B

        # --- اعتبارسنجی موج AB ---
        if bars_AB < ab_min_bars or bars_AB > ab_max_bars:
            funnel['03_Reject_AB_Bars_OutOfRange'] += 1
            continue

        ab_leg_is_up = B['price'] > A['price']

        if bars_AB == 2:
            # قانون PDF: «اگر موج AB شامل دو کندل باشد و هر کندل بزرگتر از مقدار
            # ATR آن تایم‌فریم باشد می‌توان با تخفیف آن را تایید کرد.»
            # این یعنی هر دو کندلِ تشکیل‌دهنده باید *تک‌به‌تک* رنجشون بزرگ‌تر از
            # ATR (با ضریب تخفیف) باشه — نه فقط مجموع/رنج کلی لگ. یک لگ با یک
            # کندل غول‌آسا و یک کندل ریز/دوجی نباید قبول بشه حتی اگه رنج کلی‌اش
            # بزرگ باشه.
            per_candle_ok = True
            for c_idx in (pivot_idx_A + 1, pivot_idx_B):
                atr_c = atr[c_idx]
                if np.isnan(atr_c) or atr_c <= 0:
                    per_candle_ok = False
                    break
                candle_range = high_arr[c_idx] - low_arr[c_idx]
                # 🔧 اصلاح‌شده: ضریب تخفیف قبلاً hardcoded=0.8 بود؛ الان با
                # ab2_discount_mult قابل‌تنظیمه (Gate 04 مسئول ۱۸.۲٪ ریزش بود).
                if (candle_range / atr_c) < (ab_atr_mult_min * ab2_discount_mult):
                    per_candle_ok = False
                    break
            if not per_candle_ok:
                funnel['04_Reject_AB2_PerCandle_ATR_Fail'] += 1
                continue
        else:
            # قانون PDF: «موج AB باید شامل حداقل سه کندل ... باشد و حداکثر
            # می‌تواند یک کندل نویز داشته باشد.» کندل نویز یعنی کندلی که یا در
            # جهت مخالفِ کل موج AB بسته شده، یا بدنه‌اش به‌طرز چشمگیری کوچک‌تر
            # از رنجش هست (بی‌خاصیت/دوجی). این شرط قبلاً به‌کل چک نمی‌شد.
            noise_count = 0
            for c_idx in range(pivot_idx_A + 1, pivot_idx_B + 1):
                body_c = close_arr[c_idx] - open_arr[c_idx]
                rng_c = high_arr[c_idx] - low_arr[c_idx]
                is_opposite = (body_c < 0) if ab_leg_is_up else (body_c > 0)
                is_indecisive = (rng_c <= 0) or (abs(body_c) / rng_c < 0.15)
                if is_opposite or is_indecisive:
                    noise_count += 1
            # 🔧 اصلاح‌شده (بر اساس قیف تشخیصی): سقفِ ثابتِ «حداکثر ۱ کندل نویز»
            # برای *هر* طول موجی، این Gate رو به تنهایی مسئول ۲۴.۸٪ از کل ریزش
            # کرده بود. منطقاً هم درست نیست: موج ۳کندلی و موج ۹کندلی نباید
            # سقفِ نویزِ یکسان داشته باشن. الان نسبت به طول موج مقیاس می‌گیریم:
            # حداقل ۱ کندل نویز همیشه مجازه، و به ازای هر floor(1/ab_noise_fraction_max)
            # کندلِ اضافه، یک کندل نویز بیشتر مجاز می‌شه.
            max_noise_allowed = max(1, int(bars_AB * ab_noise_fraction_max))
            if noise_count > max_noise_allowed:
                funnel['05_Reject_AB_TooMuchNoise'] += 1
                continue

            if bars_AB == 3:
                if ab_atr_ratio < ab_atr_mult_min:
                    funnel['06_Reject_AB3_ATR_Ratio_TooSmall'] += 1
                    continue
            else:
                # بیش از ۳ کندل: باید بین ۲۰۰٪ تا ۵۰۰٪ ATR باشد وگرنه یا متعلق
                # به تایم‌فریم بالاتر است یا ستاپ فییل شده.
                if not (ab_extended_min <= ab_atr_ratio <= ab_extended_max):
                    funnel['07_Reject_AB_Extended_ATR_Ratio_OutOfBand'] += 1
                    continue

        ab_is_bullish = ab_leg_is_up   # AB صعودی -> ستاپ ریورسال فروش در B
        # سناریوی SELL: AB صعودی (A=کف، B=سقف)؛ سناریوی BUY: AB نزولی (A=سقف، B=کف)
        dir_signal = -1 if ab_is_bullish else 1

        search_start = B['confirm_idx'] + 1   # 🛡️ فقط از لحظه‌ای که B رسماً تایید شده
        if search_start >= n_bars - 2:
            funnel['08_Reject_SearchStart_OutOfBounds'] += 1
            continue

        # --- ردیابی BC + باکس نقدینگی، بار به بار، کاملاً علّی ---
        bc_extreme = B['price']     # کف/سقف در حال شکل‌گیریِ BC
        bc_bars_count = 0
        c_price = None
        setup_failed = False
        cand_idx = None

        max_bc_scan = min(search_start + int(signal_search_bars) + int(bc_min_bars) + 50, n_bars - 2)

        for j in range(search_start, max_bc_scan):
            bc_bars_count += 1

            if ab_is_bullish:
                # BC اصلاح رو به پایین از B
                bc_extreme = min(bc_extreme, low_arr[j])
                retrace_ratio = (B['price'] - bc_extreme) / ab_range
                # فِیل: شدو حتی کف A رو بزنه
                if low_arr[j] < A['price']:
                    setup_failed = True
                    funnel['09_Reject_BC_ShadowPastA'] += 1
                    break
                # فِیل: بدنه بیش از ۵۰٪ اصلاح کلوز بده
                if close_arr[j] < (B['price'] - 0.5 * ab_range):
                    setup_failed = True
                    funnel['10_Reject_BC_CloseBeyond50pct'] += 1
                    break
            else:
                bc_extreme = max(bc_extreme, high_arr[j])
                retrace_ratio = (bc_extreme - B['price']) / ab_range
                if high_arr[j] > A['price']:
                    setup_failed = True
                    funnel['09_Reject_BC_ShadowPastA'] += 1
                    break
                if close_arr[j] > (B['price'] + 0.5 * ab_range):
                    setup_failed = True
                    funnel['10_Reject_BC_CloseBeyond50pct'] += 1
                    break

            if bc_bars_count < bc_min_bars:
                continue
            if retrace_ratio < bc_retrace_min:
                continue  # هنوز به حداقل اصلاح لازم نرسیده

            c_price = bc_extreme
            box_height = ab_range * box_scale
            if ab_is_bullish:
                box_top = c_price + box_height
            else:
                box_bottom = c_price - box_height

            # --- از این بار به بعد، منتظر تلاش برای شکست سطح B هستیم (کندل سیگنال) ---
            if retrace_ratio > bc_retrace_max:
                # اصلاح بیش از حد مجاز؛ دیگه این لگ به‌عنوان BC معتبر نیست
                setup_failed = True
                funnel['11_Reject_BC_RetraceExceedsMax'] += 1
                break

            if ab_is_bullish and close_arr[j] > B['price']:
                # کاندید کندل سیگنال (شکار نقدینگی صعودی فراتر از B)
                if close_arr[j] > box_top:
                    setup_failed = True  # شکست واقعی و قاطع؛ دیگه ریورسال نیست
                    funnel['12_Reject_BC_RealBreakoutPastBox'] += 1
                    break
                cand_idx = j
                break
            elif (not ab_is_bullish) and close_arr[j] < B['price']:
                if close_arr[j] < box_bottom:
                    setup_failed = True
                    funnel['12_Reject_BC_RealBreakoutPastBox'] += 1
                    break
                cand_idx = j
                break
        else:
            setup_failed = True  # از سقف اسکن رد شدیم بدون پیدا کردن کاندید
            funnel['13_Reject_BC_ScanExhausted_NoSweep'] += 1

        if setup_failed or c_price is None or cand_idx is None:
            continue

        box_height = ab_range * box_scale
        if ab_is_bullish:
            box_top = c_price + box_height
            box_bottom = B['price'] - 0.5 * ab_range  # حد پایینِ منطقه‌ی معتبر برای مرجع بصری
        else:
            box_bottom = c_price - box_height
            box_top = B['price'] + 0.5 * ab_range

        # --- بررسی کیفیت کندل سیگنال (منفرد؛ اگر رد شد، تلاش با ترکیب FL) ---
        def check_candle_span(i0, i1):
            """ترکیب کندل‌های i0..i1 (شامل) به یک کندل مصنوعی و بررسی کیفیت."""
            o = open_arr[i0]
            c = close_arr[i1]
            h = high_arr[i0:i1 + 1].max()
            l = low_arr[i0:i1 + 1].min()
            atr_ref = atr[i1]
            ok = _candle_quality(o, h, l, c, atr_ref, signal_body_ratio_min,
                                  signal_wick_reject_ratio, signal_atr_mult, ab_is_bullish)
            if not ok:
                return False, None
            if ab_is_bullish and c <= B['price']:
                return False, None
            if (not ab_is_bullish) and c >= B['price']:
                return False, None
            return True, (h, l)

        sig_span = None
        ok, hl = check_candle_span(cand_idx, cand_idx)
        if ok:
            sig_span = (cand_idx, cand_idx, hl)
        elif allow_fl_candle:
            for extra in (1, 2):
                i1 = cand_idx + extra
                if i1 >= n_bars - 2:
                    break
                if ab_is_bullish and high_arr[i1] <= high_arr[cand_idx]:
                    continue  # قانون FL: های کندل دوم/سوم باید های کندل اول رو بزنه
                if (not ab_is_bullish) and low_arr[i1] >= low_arr[cand_idx]:
                    continue
                ok2, hl2 = check_candle_span(cand_idx, i1)
                if ok2:
                    sig_span = (cand_idx, i1, hl2)
                    break

        if sig_span is None:
            funnel['14_Reject_SignalCandle_QualityFail'] += 1
            continue

        sig_start, sig_end, (sig_high, sig_low) = sig_span
        sl_ref_extreme_high = sig_high
        sl_ref_extreme_low = sig_low

        # --- جست‌وجوی کندل تایید/تریگر ---
        trig_idx = None
        confirm_end = min(sig_end + 1 + int(confirm_search_bars), n_bars - 2)
        box_violation = False
        for t in range(sig_end + 1, confirm_end):
            # کندل‌های میانی می‌تونن بالاتر (برای سناریوی SELL) از کندل سیگنال کلوز بدن،
            # اما نباید باکس نقدینگی رو نقض کنن (یعنی به‌طور قاطع فراتر از سطح
            # مرجع باکس کلوز ندن — علامتِ شکست واقعیِ ساختار به‌جای ریورسال).
            if ab_is_bullish:
                sl_ref_extreme_high = max(sl_ref_extreme_high, high_arr[t])
                if close_arr[t] > box_top:
                    box_violation = True
                    break
                rng_t = high_arr[t] - low_arr[t]
                atr_t = atr[t]
                if (not np.isnan(atr_t)) and atr_t > 0 and (rng_t / atr_t) >= confirm_atr_mult and close_arr[t] < sig_low:
                    trig_idx = t
                    break
            else:
                sl_ref_extreme_low = min(sl_ref_extreme_low, low_arr[t])
                if close_arr[t] < box_bottom:
                    box_violation = True
                    break
                rng_t = high_arr[t] - low_arr[t]
                atr_t = atr[t]
                if (not np.isnan(atr_t)) and atr_t > 0 and (rng_t / atr_t) >= confirm_atr_mult and close_arr[t] > sig_high:
                    trig_idx = t
                    break

        if box_violation or trig_idx is None:
            if box_violation:
                funnel['15_Reject_Confirm_BoxViolation'] += 1
            else:
                funnel['16_Reject_Confirm_NoTriggerFound'] += 1
            continue

        # --- شرط صریحِ واگرایی قیمتی AB/CD ---
        # ⚠️ نکته‌ی مهم: خودِ PDF در این مورد دو عبارت به‌ظاهر متناقض دارد:
        #   خط اول متن می‌گوید «موج AB باید از موج CD ... بزرگ‌تر باشد» (AB > CD)
        #   ولی دو جای دیگر (بخش باکس نقدینگی + بخش کندل‌های میانی) صراحتاً و با
        #   نماد ریاضی می‌نویسد «شرط CD>AB» / «نکته موج CD>AB نباید نقض شود».
        # چون این تناقض در خودِ سند اصلی وجود دارد، این چک را به‌صورت یک حالتِ
        # قابل‌انتخاب (cd_ab_mode) پیاده کردیم تا هر دو خوانش قابل تست باشند؛
        # پیش‌فرض روی «CD > AB» است چون دو بار با نماد صریح تکرار شده و از نظر
        # منطق معاملاتی هم با هدفِ این ستاپ (یک ریورسال قدرتمندتر از ایمپالسِ
        # اولیه، با تی‌پی ۲.۵ تا ۳ آر) سازگارتر است.
        # نقطه‌ی D = نهایی‌ترین اکسترمم رسیده‌شده در طول کندل سیگنال/FL و کندل‌های
        # میانیِ تا کندل تایید (یعنی همان sl_ref_extreme که در ادامه برای استاپ هم استفاده می‌شود).
        d_price = sl_ref_extreme_high if ab_is_bullish else sl_ref_extreme_low
        cd_range = abs(d_price - c_price)

        if cd_ab_mode == "AB > CD (طبق خط اول PDF)":
            if not (ab_range > cd_range):
                funnel['17_Reject_CD_AB_Ratio_Fail'] += 1
                continue
        else:  # "CD > AB (طبق نقاط تکرارشده در PDF)" - پیش‌فرض
            if not (cd_range > ab_range):
                funnel['17_Reject_CD_AB_Ratio_Fail'] += 1
                continue

        entry_idx = trig_idx + 1
        if entry_idx >= n_bars - 1:
            funnel['18_Reject_Entry_OutOfBounds'] += 1
            continue
        entry_price = open_arr[entry_idx]
        atr_at_entry = atr[trig_idx] if not np.isnan(atr[trig_idx]) and atr[trig_idx] > 0 else atr_at_B
        buffer_amt = sl_buffer_atr_mult * atr_at_entry

        if ab_is_bullish:  # ستاپ فروش
            stop = sl_ref_extreme_high + buffer_amt
            risk = stop - entry_price
        else:  # ستاپ خرید
            stop = sl_ref_extreme_low - buffer_amt
            risk = entry_price - stop

        if risk <= 0:
            funnel['19_Reject_NonPositiveRisk'] += 1
            continue

        funnel['20_ACCEPTED_Raw_Signal'] += 1
        signals.append({
            'idx': entry_idx, 'time': df.index[entry_idx], 'dir': dir_signal,
            'A_time': df.index[pivot_idx_A], 'B_time': df.index[pivot_idx_B],
            'C_price': c_price, 'D_price': d_price, 'signal_time': df.index[sig_end], 'trigger_time': df.index[trig_idx],
            'entry_price': entry_price, 'M1_SL': stop, 'risk_raw': risk,
            'Target_Class': 0,
            'PS_AB_Range_ATR': ab_atr_ratio, 'PS_AB_Bars': bars_AB,
            'PS_BC_Retrace_Ratio': retrace_ratio, 'PS_BC_Bars': bc_bars_count,
            'PS_CD_AB_Ratio': cd_range / ab_range,
            'PS_Bars_Signal_To_Trigger': trig_idx - sig_end,
            'PS_FL_Candle_Used': int(sig_end != sig_start),
            'PS_Box_Top': box_top, 'PS_Box_Bottom': box_bottom,
        })

    signals.sort(key=lambda s: s['idx'])
    return signals, funnel

# =============================================================================
# 7. موتور تصویرسازی (Visual Generator — سبک یکسان با بقیه‌ی سلول‌های پروژه)
# =============================================================================
def generate_chart_pivot_settlement(df, sig_dict, rr, max_bars, save_path):
    try:
        t0_time = sig_dict['time']
        loc = df.index.get_loc(t0_time)
        start = max(0, loc - 40)
        end = min(len(df), loc + int(max_bars) + 15)
        window = df.iloc[start:end].copy()
        window['x'] = np.arange(len(window))
        t0_x = window.index.get_loc(t0_time)

        fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0b0f19')
        ax.set_facecolor('#0b0f19')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#333')

        up = window[window['Close'] >= window['Open']]
        down = window[window['Close'] < window['Open']]
        ax.bar(up['x'], up['Close'] - up['Open'], bottom=up['Open'], color='#00ffcc', width=0.6, zorder=3)
        ax.vlines(up['x'], up['Low'], up['High'], color='#00ffcc', linewidth=1, zorder=2)
        ax.bar(down['x'], down['Open'] - down['Close'], bottom=down['Close'], color='#ff3366', width=0.6, zorder=3)
        ax.vlines(down['x'], down['Low'], down['High'], color='#ff3366', linewidth=1, zorder=2)

        # ناحیه‌ی باکس نقدینگی
        if window.index.min() <= sig_dict['signal_time'] <= window.index.max():
            ax.axhspan(sig_dict['PS_Box_Bottom'], sig_dict['PS_Box_Top'], color='#00f2ff', alpha=0.10, zorder=0)

        entry_price = sig_dict['entry_price']
        sl = sig_dict['M1_SL']
        risk = sig_dict['risk_raw']
        tp = entry_price - rr * risk if sig_dict['dir'] == -1 else entry_price + rr * risk

        target_class = sig_dict['Target_Class']
        entry_color = '#00ff88' if target_class == 1 else '#ff0055'

        ax.hlines(entry_price, t0_x, t0_x + max_bars, color=entry_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.hlines(sl, t0_x, t0_x + max_bars, color='#ff3366', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.hlines(tp, t0_x, t0_x + max_bars, color='#00ffcc', linestyle='-', linewidth=2, alpha=0.8)
        ax.scatter([t0_x], [entry_price], color=entry_color, marker='v' if sig_dict['dir'] == -1 else '^', s=200, zorder=5)

        class_names = {0: "LOSS / TIMEOUT (Class 0)", 1: "CLEAN WIN (Class 1)"}
        dir_text = "PIVOT SETTLEMENT SELL" if sig_dict['dir'] == -1 else "PIVOT SETTLEMENT BUY"
        ax.set_title(f"Pair: {sig_dict.get('pair','')} | Mode: Pivot Settlement v1.0 ({dir_text}) | Result: {class_names[target_class]}",
                     color=entry_color, fontsize=12, fontweight='bold')

        plt.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    except Exception:
        plt.close('all')

# =============================================================================
# 8. منطق اصلی پردازش (The Pivot Settlement Master Forge)
# =============================================================================
def process_labeling_pivot_settlement(datasets, swing_n, ab_min_bars, ab_max_bars, ab_atr_mult_min,
                                       ab_extended_min, ab_extended_max, bc_min_bars,
                                       bc_retrace_min, bc_retrace_max, box_scale,
                                       signal_search_bars, signal_atr_mult, signal_body_ratio_min,
                                       signal_wick_reject_ratio, allow_fl_candle,
                                       confirm_search_bars, confirm_atr_mult, sl_buffer_atr_mult,
                                       cd_ab_mode, rr, spread_raw, max_bars, use_trend_gate, trend_gate_mode,
                                       ab_noise_fraction_max, ab2_discount_mult,
                                       progress=gr.Progress()):
    if not datasets or "No Data Found" in datasets:
        return "❌ دیتاست یافت نشد.", None, None, None, None

    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)

    final_message = f"🚀 عملیات استخراج سیگنال‌های «پیوت تسویه» روی {len(datasets)} نماد آغاز شد...\n"
    last_tail_df = pd.DataFrame()

    global_stats = {
        'Total Pivot Settlement Signals': 0,
        'Class 1 (Winning Signal)': 0,
        'Class 0 (Losing Signal)': 0
    }
    global_funnel = Counter()

    for idx, dataset_name in enumerate(datasets):
        pair = dataset_name.split('_')[0]
        base_prog = idx / len(datasets)
        step_prog = 1.0 / len(datasets)

        features_path = os.path.join(DATA_DIR, f"{dataset_name}_Features.parquet")
        progress(base_prog + (0.05 * step_prog), desc=f"[{pair}] 📥 Loading Features...")
        df = pd.read_parquet(features_path)

        progress(base_prog + (0.20 * step_prog), desc=f"[{pair}] 🧭 Scanning AB -> BC -> Box -> Signal -> Trigger...")
        signals, funnel = build_pivot_settlement_signals(
            df, int(swing_n), int(ab_min_bars), int(ab_max_bars), float(ab_atr_mult_min),
            float(ab_extended_min), float(ab_extended_max), int(bc_min_bars),
            float(bc_retrace_min), float(bc_retrace_max), float(box_scale),
            int(signal_search_bars), float(signal_atr_mult), float(signal_body_ratio_min),
            float(signal_wick_reject_ratio), bool(allow_fl_candle),
            int(confirm_search_bars), float(confirm_atr_mult), float(sl_buffer_atr_mult),
            cd_ab_mode, float(ab_noise_fraction_max), float(ab2_discount_mult)
        )
        for s in signals:
            s['pair'] = pair
        global_funnel.update(funnel)

        # --- فیلتر اختیاری هم‌جهتی با روند بلندمدت (Optional Trend Confluence Gate) ---
        if use_trend_gate and signals:
            if 'EMA_Stack_Score' in df.columns:
                filtered = []
                for s in signals:
                    try:
                        trend_val = df['EMA_Stack_Score'].loc[s['B_time']]
                    except Exception:
                        continue
                    if pd.isna(trend_val):
                        continue
                    # dir=-1 (SELL) یعنی AB صعودی بوده (روند صعودی که داره سقف می‌سازه)
                    prevailing_up = trend_val > 0
                    if trend_gate_mode == "خلاف روند (Counter-Trend Reversal)":
                        # ریورسال واقعی: روند غالب باید هم‌جهتِ خودِ AB (یعنی مخالفِ جهت معامله) باشد
                        keep = (prevailing_up and s['dir'] == -1) or ((not prevailing_up) and s['dir'] == 1)
                    else:
                        # با روند بزرگ‌تر: معامله باید هم‌جهتِ روند بلندمدت باشد
                        keep = (prevailing_up and s['dir'] == 1) or ((not prevailing_up) and s['dir'] == -1)
                    if keep:
                        filtered.append(s)
                signals = filtered
            else:
                final_message += f"⚠️ [{pair}] فیلتر روند فعال بود ولی ستون EMA_Stack_Score در دیتاست نبود؛ فیلتر نادیده گرفته شد.\n"

        total_signals = len(signals)
        global_stats['Total Pivot Settlement Signals'] += total_signals

        results = []
        tick_manager = TickManager(DATA_DIR, pair)

        for s_idx, sig in enumerate(signals):
            if total_signals > 0 and s_idx % max(1, total_signals // 10) == 0:
                progress(base_prog + (0.4 * step_prog) + (0.4 * step_prog * (s_idx / max(1, total_signals))),
                         desc=f"[{pair}] ⏱️ Tick Racing {s_idx}/{total_signals}")

            entry_price = sig['entry_price']
            risk = sig['risk_raw']
            if sig['dir'] == -1:
                tp = entry_price - rr * risk
            else:
                tp = entry_price + rr * risk
            sig['M1_TP'] = tp
            sig['max_bars'] = max_bars

            entry_t = sig['time']
            entry_loc = df.index.get_loc(entry_t)
            limit_idx = min(entry_loc + int(max_bars), len(df) - 1)
            limit_time = df.index[limit_idx]

            future_ticks_series = tick_manager.get_ticks_series(entry_t, limit_time)

            if len(future_ticks_series) > 0:
                status, _ = simulate_race_ps(
                    future_ticks_series.values, sig['dir'], sig['M1_SL'], sig['M1_TP'], spread_raw
                )
                is_winner = 1 if status == 1 else 0  # TP=1، SL یا Timeout=0
            else:
                is_winner = 0

            sig['Target_Class'] = is_winner
            results.append(sig)

        tick_manager.clear_memory()

        progress(base_prog + (0.85 * step_prog), desc=f"[{pair}] 🧠 Packing Matrix...")

        meta_cols = ['PS_AB_Range_ATR', 'PS_AB_Bars', 'PS_BC_Retrace_Ratio', 'PS_BC_Bars',
                     'PS_CD_AB_Ratio', 'PS_Bars_Signal_To_Trigger', 'PS_FL_Candle_Used']
        for col in ['entry_price', 'M1_SL', 'M1_TP', 'max_bars'] + meta_cols:
            df[col] = 0.0
        df['Target_Class'] = 0
        df['Signal_Dir'] = 0.0

        for sig in results:
            df.at[sig['time'], 'Target_Class'] = sig['Target_Class']
            df.at[sig['time'], 'Signal_Dir'] = float(sig['dir'])
            for col in ['entry_price', 'M1_SL', 'M1_TP', 'max_bars'] + meta_cols:
                df.at[sig['time'], col] = sig[col]

        df_filtered = df[df['Signal_Dir'] != 0.0].copy()

        progress(base_prog + (0.90 * step_prog), desc=f"[{pair}] 📸 Generating Live Horizon Plots...")
        if len(results) > 0:
            num_charts = min(20, len(results))
            indices = np.linspace(0, len(results) - 1, num_charts, dtype=int)
            sampled_sigs = [results[i] for i in indices]
            for count, sig_dict in enumerate(sampled_sigs):
                save_path = f"{IMG_DIR}/Chart_{pair}_{count:02d}_Class_{sig_dict['Target_Class']}.png"
                generate_chart_pivot_settlement(df, sig_dict, rr, max_bars, save_path)

        out_path = os.path.join(DATA_DIR, f"{dataset_name}_Labeled.parquet")
        df_filtered.to_parquet(out_path, compression='snappy')

        c1 = sum(1 for s in results if s['Target_Class'] == 1)
        c0 = sum(1 for s in results if s['Target_Class'] == 0)
        global_stats['Class 1 (Winning Signal)'] += c1
        global_stats['Class 0 (Losing Signal)'] += c0

        final_message += f"✅ دیتاست {pair} تکمیل شد. (Signals: {total_signals} | Wins: {c1} | Losses: {c0})\n"
        last_tail_df = df_filtered.tail(100).reset_index()
        gc.collect()

    progress(0.95, desc="📦 Generating Dynamic MetaData...")

    metadata = {
        "num_classes": 2,
        "class_mapping": {0: "Loss/Timeout (Class 0)", 1: "Clean Win (Class 1)"},
        "strategy": "Pivot_Settlement_Setup",
        "features_version": "v25.0"
    }
    with open(os.path.join(DATA_DIR, "dataset_metadata.json"), "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    zip_path = shutil.make_archive(os.path.join(DATA_DIR, "HIPO_PivotSettlement_Visuals"), 'zip', IMG_DIR) if len(os.listdir(IMG_DIR)) > 0 else None

    stats_df = pd.DataFrame({'Category': list(global_stats.keys()), 'Count': list(global_stats.values())})
    total = global_stats['Total Pivot Settlement Signals']
    stats_df['Percentage %'] = (stats_df['Count'] / total * 100).round(2) if total > 0 else 0.0

    final_message += "\n🏆 پایان استخراج سیگنال! دیتای ۲-کلاسه‌ی «پیوت تسویه» ذخیره شد و متادیتا برای آموزش هوش مصنوعی آماده‌ست."

    # --- جدول قیف تشخیصی (Diagnostic Rejection Funnel) ---
    # هدف: نشون بدیم دقیقاً کدوم شرط (Gate) بیشترین کاندیدها رو حذف می‌کنه،
    # تا به‌جای حدس‌زدن، بر اساس داده‌ی واقعی تصمیم بگیریم کدوم پارامتر رو شل کنیم.
    total_candidates = global_funnel.get('00_Total_AB_Candidates (zigzag pairs)', 0)
    funnel_rows = []
    for gate_name in sorted(global_funnel.keys()):
        count = global_funnel[gate_name]
        pct_of_total = (count / total_candidates * 100) if total_candidates > 0 else 0.0
        funnel_rows.append({'Gate': gate_name, 'Count': count, '% of Total AB Candidates': round(pct_of_total, 2)})
    funnel_df = pd.DataFrame(funnel_rows)

    return final_message, last_tail_df, stats_df, zip_path, funnel_df

# =============================================================================
# 8B. موتور اپتیمایزیشن ژنتیک پارامترها (Genetic Algorithm Optimizer)
# -----------------------------------------------------------------------------
# ⚠️ توجه مهم: این بخش هیچ تغییری در منطق کشف سیگنال (build_pivot_settlement_signals)
# یا در تابع اصلی پردازش (process_labeling_pivot_settlement) نمی‌دهد — هر دو
# تابع دقیقاً همان‌طور که در بالا تعریف شده‌اند، بدون هیچ تغییری، توسط این بخش
# فراخوانی می‌شوند.
#
# چرا ژنتیک به‌جای جست‌وجوی تصادفیِ خالص؟
# جست‌وجوی تصادفی (نسخه‌ی قبلی) هیچ حافظه‌ای از ترکیب‌های خوب ندارد؛ هر ترایال
# کاملاً مستقل و کور است، برای همین روی یک فضای ۲۳بعدی با شرط‌های AND تودرتو
# تقریباً همیشه به نواحی بی‌ارزش (مثل ۸٪ وین‌ریت) برخورد می‌کند. الگوریتم ژنتیک
# در هر نسل، بهترین‌ها را نگه می‌دارد (Elitism)، از تلفیق دو والدِ خوب فرزند
# می‌سازد (Crossover) و با جهش کنترل‌شده (Mutation) فضای اطراف را کاوش می‌کند؛
# دقیقاً همان اصلی که موتور Optimizer متاتریدر (Strategy Tester) هم بر پایه‌ی
# آن کار می‌کند: نسل به نسل، جمعیت به‌سمت راه‌حل‌های بهتر «تکامل» پیدا می‌کند.
#
# هدف بهینه‌سازی (طبق خواسته‌ی صریح): بالا بردن هم *تعداد* معاملاتِ برنده
# (کلاس ۱) و هم *درصدِ* آن‌ها — نه صرفاً نزدیک‌کردن به توازن ۵۰/۵۰. تابع فیتنس
# = (تعداد بردها) × (درصد وین‌ریت) — این معیار هم‌زمان جریمه می‌کند اگر تعداد
# بردها کم باشد (even at high %) و هم اگر درصد پایین باشد (even at high count)؛
# تنها با بالا رفتن *همزمان* هر دو، فیتنس واقعاً بالا می‌رود. حداقل تعداد کل
# نمونه (برای جلوگیری از نتیجه‌ی تصادفی/اورفیت‌شده روی دیتای خیلی کم) هم‌چنان
# به‌صورت قید سخت اعمال می‌شود.
# =============================================================================

_OPT_PARAM_SPACE = {
    'swing_n':                  (1, 10, 'int'),
    'ab_min_bars':               (2, 10, 'int'),
    'ab_max_bars':                (3, 20, 'int'),
    'ab_atr_mult_min':           (0.5, 5.0, 'float'),
    'ab_extended_min':           (1.0, 6.0, 'float'),
    'ab_extended_max':           (2.0, 10.0, 'float'),
    'bc_min_bars':                (1, 15, 'int'),
    'bc_retrace_min':            (0.05, 0.6, 'float'),
    'bc_retrace_max':            (0.2, 0.8, 'float'),
    'box_scale':                  (0.5, 2.0, 'float'),
    'signal_search_bars':         (5, 150, 'int'),
    'signal_atr_mult':           (0.1, 2.0, 'float'),
    'signal_body_ratio_min':     (0.0, 1.0, 'float'),
    'signal_wick_reject_ratio':  (0.3, 3.0, 'float'),
    'allow_fl_candle':            (0, 1, 'bool'),
    'confirm_search_bars':        (1, 50, 'int'),
    'confirm_atr_mult':          (0.1, 2.0, 'float'),
    'sl_buffer_atr_mult':        (0.0, 2.0, 'float'),
    'ab_noise_fraction_max':     (0.1, 0.6, 'float'),
    'ab2_discount_mult':         (0.3, 1.0, 'float'),
    'rr':                         (1.0, 6.0, 'float'),
    'max_bars':                   (10, 300, 'int'),
    'use_trend_gate':             (0, 1, 'bool'),
}
_OPT_CD_AB_CHOICES = ["CD > AB (طبق نقاط تکرارشده در PDF)", "AB > CD (طبق خط اول PDF)"]
_OPT_TREND_MODE_CHOICES = ["خلاف روند (Counter-Trend Reversal)", "هم‌جهت روند (With-Trend Continuation)"]


def _opt_fix_constraints(p):
    """رفع ناسازگاری بازه‌ها (min باید <= max باشد) — بعد از sample/crossover/mutate صدا زده می‌شود."""
    if p['ab_min_bars'] > p['ab_max_bars']:
        p['ab_min_bars'], p['ab_max_bars'] = p['ab_max_bars'], p['ab_min_bars']
    if p['ab_extended_min'] > p['ab_extended_max']:
        p['ab_extended_min'], p['ab_extended_max'] = p['ab_extended_max'], p['ab_extended_min']
    if p['bc_retrace_min'] > p['bc_retrace_max']:
        p['bc_retrace_min'], p['bc_retrace_max'] = p['bc_retrace_max'], p['bc_retrace_min']
    return p


def _opt_sample_params(rng):
    """ساخت یک فرد کاملاً تصادفی — برای جمعیت اولیه و برای 'مهاجرِ تصادفی' هر نسل."""
    p = {}
    for name, (lo, hi, typ) in _OPT_PARAM_SPACE.items():
        if typ == 'int':
            p[name] = int(rng.integers(lo, hi + 1))
        elif typ == 'bool':
            p[name] = bool(rng.integers(0, 2))
        else:
            p[name] = float(rng.uniform(lo, hi))
    p['cd_ab_mode'] = _OPT_CD_AB_CHOICES[int(rng.integers(0, 2))]
    p['trend_gate_mode'] = _OPT_TREND_MODE_CHOICES[int(rng.integers(0, 2))]
    return _opt_fix_constraints(p)


def _opt_crossover(parent_a, parent_b, rng):
    """کراس‌آور یکنواخت: هر ژن با احتمال ۵۰٪ از یکی از دو والد به ارث می‌رسد."""
    child = {}
    for name in _OPT_PARAM_SPACE.keys():
        child[name] = parent_a[name] if rng.random() < 0.5 else parent_b[name]
    child['cd_ab_mode'] = parent_a['cd_ab_mode'] if rng.random() < 0.5 else parent_b['cd_ab_mode']
    child['trend_gate_mode'] = parent_a['trend_gate_mode'] if rng.random() < 0.5 else parent_b['trend_gate_mode']
    return _opt_fix_constraints(child)


def _opt_mutate(individual, rng, mutation_rate):
    """
    جهش ژن‌به‌ژن: به‌ازای هر پارامتر، با احتمال mutation_rate یا با یک تکانِ
    گاوسیِ کوچک اطراف مقدار فعلی کاوش می‌شود (Local Search) یا به‌طور کامل از نو
    از بازه‌ی مجاز نمونه‌برداری می‌شود (Global Jump) — ترکیبی که هم دقیق کاوش
    می‌کند هم از گیر افتادن در یک نقطه‌ی محلی جلوگیری می‌کند.
    """
    ind = dict(individual)
    for name, (lo, hi, typ) in _OPT_PARAM_SPACE.items():
        if rng.random() >= mutation_rate:
            continue
        if typ == 'int':
            if rng.random() < 0.6:
                span = max(1, int((hi - lo) * 0.15))
                ind[name] = int(np.clip(ind[name] + rng.integers(-span, span + 1), lo, hi))
            else:
                ind[name] = int(rng.integers(lo, hi + 1))
        elif typ == 'bool':
            ind[name] = not ind[name]
        else:
            if rng.random() < 0.6:
                span = hi - lo
                ind[name] = float(np.clip(ind[name] + rng.normal(0, span * 0.15), lo, hi))
            else:
                ind[name] = float(rng.uniform(lo, hi))
    if rng.random() < mutation_rate:
        ind['cd_ab_mode'] = _OPT_CD_AB_CHOICES[int(rng.integers(0, 2))]
    if rng.random() < mutation_rate:
        ind['trend_gate_mode'] = _OPT_TREND_MODE_CHOICES[int(rng.integers(0, 2))]
    return _opt_fix_constraints(ind)


def _opt_tournament_select(evaluated, k, rng):
    """انتخاب مسابقه‌ای: k فرد تصادفی از جمعیت برمی‌داریم، برنده (بالاترین فیتنس) والد می‌شود."""
    idxs = rng.integers(0, len(evaluated), size=k)
    best = max(idxs, key=lambda i: evaluated[i][0])
    return evaluated[best][1]


def _opt_preload_datasets(datasets):
    """هر دیتاست فقط یک‌بار از دیسک لود می‌شود تا در طول صدها ارزیابی دوباره خوانده نشود."""
    loaded = []
    for dataset_name in datasets:
        pair = dataset_name.split('_')[0]
        features_path = os.path.join(DATA_DIR, f"{dataset_name}_Features.parquet")
        if not os.path.exists(features_path):
            continue
        df = pd.read_parquet(features_path)
        tick_manager = TickManager(DATA_DIR, pair)
        loaded.append({'pair': pair, 'dataset_name': dataset_name, 'df': df, 'tm': tick_manager})
    return loaded


def _opt_evaluate_config(loaded_datasets, params, spread_raw):
    """
    اجرای فوق‌سریعِ یک فرد — build_pivot_settlement_signals و simulate_race_ps را
    عیناً (بدون‌تغییر) صدا می‌زند، فقط بدون ذخیره‌ی پارکت/نمودار، صرفاً برای
    شمارش تعداد سیگنال و تعداد بردها.
    """
    total_signals = 0
    total_wins = 0
    for item in loaded_datasets:
        df, tick_manager = item['df'], item['tm']
        signals, _funnel = build_pivot_settlement_signals(
            df, params['swing_n'], params['ab_min_bars'], params['ab_max_bars'], params['ab_atr_mult_min'],
            params['ab_extended_min'], params['ab_extended_max'], params['bc_min_bars'],
            params['bc_retrace_min'], params['bc_retrace_max'], params['box_scale'],
            params['signal_search_bars'], params['signal_atr_mult'], params['signal_body_ratio_min'],
            params['signal_wick_reject_ratio'], params['allow_fl_candle'],
            params['confirm_search_bars'], params['confirm_atr_mult'], params['sl_buffer_atr_mult'],
            params['cd_ab_mode'], params['ab_noise_fraction_max'], params['ab2_discount_mult']
        )

        # --- عیناً همان فیلتر اختیاری هم‌جهتی با روند از process_labeling_pivot_settlement ---
        if params['use_trend_gate'] and signals:
            if 'EMA_Stack_Score' in df.columns:
                filtered = []
                for s in signals:
                    try:
                        trend_val = df['EMA_Stack_Score'].loc[s['B_time']]
                    except Exception:
                        continue
                    if pd.isna(trend_val):
                        continue
                    prevailing_up = trend_val > 0
                    if params['trend_gate_mode'] == "خلاف روند (Counter-Trend Reversal)":
                        keep = (prevailing_up and s['dir'] == -1) or ((not prevailing_up) and s['dir'] == 1)
                    else:
                        keep = (prevailing_up and s['dir'] == 1) or ((not prevailing_up) and s['dir'] == -1)
                    if keep:
                        filtered.append(s)
                signals = filtered

        if not signals:
            continue

        for sig in signals:
            entry_price = sig['entry_price']
            risk = sig['risk_raw']
            tp = entry_price - params['rr'] * risk if sig['dir'] == -1 else entry_price + params['rr'] * risk
            entry_t = sig['time']
            try:
                entry_loc = df.index.get_loc(entry_t)
            except Exception:
                continue
            limit_idx = min(entry_loc + int(params['max_bars']), len(df) - 1)
            limit_time = df.index[limit_idx]
            future_ticks_series = tick_manager.get_ticks_series(entry_t, limit_time)
            if len(future_ticks_series) > 0:
                status, _ = simulate_race_ps(future_ticks_series.values, sig['dir'], sig['M1_SL'], tp, spread_raw)
                is_winner = 1 if status == 1 else 0
            else:
                is_winner = 0
            total_signals += 1
            total_wins += is_winner

    return total_signals, total_wins



def _fast_feature_quality_score(signal_rows, max_feat_probe=25):
    """امتیاز 0-100: فیچرها چقدر Win/Loss این لیبل را جدا می‌کنند؟"""
    if not signal_rows or len(signal_rows) < 25:
        return 0.0
    try:
        sdf = pd.DataFrame(signal_rows)
    except Exception:
        return 0.0
    if 'Target_Class' not in sdf.columns:
        return 0.0
    y = sdf['Target_Class'].astype(int).values
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return 0.0
    drop = {
        "Open", "High", "Low", "Close", "Volume", "Tick_Up_Count", "Tick_Down_Count",
        "Bid", "Ask", "Target_Class", "Signal_Dir", "entry_price", "M1_SL", "M1_TP",
        "max_bars", "pair", "Raw_ATR",
    }
    cols = []
    for c in sdf.columns:
        if c in drop or str(c).startswith('__'):
            continue
        try:
            if not pd.api.types.is_numeric_dtype(sdf[c]):
                continue
        except Exception:
            continue
        cols.append(c)
    if not cols:
        return 0.0
    X = sdf[cols].replace([np.inf, -np.inf], np.nan).fillna(sdf[cols].median(numeric_only=True))
    scores = []
    for c in cols:
        v = X[c].values.astype(np.float64)
        if np.std(v) < 1e-12:
            continue
        r = np.corrcoef(v, y)[0, 1]
        if np.isnan(r):
            continue
        order = np.argsort(v)
        yo = y[order]
        n_pos = float(yo.sum())
        n_neg = float(len(yo) - n_pos)
        if n_pos < 1 or n_neg < 1:
            continue
        ranks = np.arange(1, len(yo) + 1, dtype=np.float64)
        auc = (ranks[yo == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = max(float(auc), 1.0 - float(auc))
        q = 0.6 * float(np.clip((auc - 0.5) / 0.15, 0, 1)) + 0.4 * float(np.clip(abs(r) / 0.25, 0, 1))
        scores.append(q)
    if not scores:
        return 0.0
    scores = sorted(scores, reverse=True)
    top = scores[:max_feat_probe]
    mean_top = float(np.mean(top))
    n_strong = sum(1 for s in scores if s >= 0.45)
    cov = min(1.0, n_strong / 8.0)
    sf = float(np.clip(np.log1p(len(y)) / np.log1p(800), 0.4, 1.0))
    return float(np.clip(100.0 * mean_top * (0.7 + 0.3 * cov) * sf, 0, 100))


def _opt_fitness(total_signals, wins, min_samples, feature_quality=0.0,
                 require_wins_gt_losses=True, min_winrate=0.50,
                 w_win_edge=0.40, w_count=0.25, w_feat=0.35):
    """
    فیتنس v5 — اهداف اصلی (طبق خواسته):
      1) Wins > Losses  (بردها بیشتر از باخت‌ها) — قید سخت اگر require_wins_gt_losses
      2) Feature_Quality_Score بالا (فیچرها بتوانند Win/Loss را جدا کنند)
      3) تعداد نمونه کافی (min_samples)

    اگر Wins <= Losses → فیتنس منفی بزرگ (رد می‌شود مگر require=False).
    """
    if total_signals <= 0:
        return -1_000_000.0

    losses = total_signals - wins
    p = wins / total_signals
    win_edge = wins - losses  # >0 یعنی بردها بیشتر

    # ---- قید سخت: بردها باید بیشتر از باخت‌ها باشند ----
    if require_wins_gt_losses and wins <= losses:
        # هنوز رتبه‌بندی نسبی برای GA: نزدیک‌تر به تساوی کمی بهتر از فاجعه
        return -500_000.0 - float(losses - wins) - (100.0 - feature_quality) * 0.05

    # ---- قید سخت: حداقل نمونه ----
    if total_signals < min_samples:
        # اگر Wins>Losses ولی نمونه کم: منفی ملایم‌تر
        bonus_edge = max(0.0, float(win_edge)) * 2.0
        return -float(min_samples - total_signals) + bonus_edge - (100.0 - feature_quality) * 0.1

    # ---- جزء لبه برد (Wins - Losses و WinRate) ----
    # تشویق قوی برای p > 0.5
    edge_ratio = win_edge / total_signals  # در [-1,1]، اینجا >0
    wr_score = 100.0 * float(np.clip((p - 0.50) / 0.20, 0, 1))  # 50%→0 ، 70%→100
    edge_count_score = 100.0 * float(np.clip(np.log1p(max(0, win_edge)) / np.log1p(max(min_samples * 0.2, 50)), 0, 1))
    win_component = 0.55 * wr_score + 0.45 * edge_count_score

    # ---- جزء تعداد ----
    count_score = 100.0 * float(np.clip(np.log1p(total_signals) / np.log1p(max(min_samples * 3, 1500)), 0, 1))
    # تعداد برد خام
    wins_score = 100.0 * float(np.clip(np.log1p(wins) / np.log1p(max(min_samples * 0.55, 200)), 0, 1))
    count_component = 0.6 * count_score + 0.4 * wins_score

    # ---- جزء کیفیت فیچر ----
    feat_component = float(np.clip(feature_quality, 0, 100))

    # نرمال وزن
    wsum = float(w_win_edge + w_count + w_feat) + 1e-9
    w1, w2, w3 = w_win_edge / wsum, w_count / wsum, w_feat / wsum
    blended = w1 * win_component + w2 * count_component + w3 * feat_component

    # هماهنگی: هر سه باید با هم بالا بروند
    harmony = 1.0 - 0.12 * (
        abs(win_component - count_component) + abs(count_component - feat_component) + abs(win_component - feat_component)
    ) / 200.0
    harmony = float(np.clip(harmony, 0.75, 1.08))

    # بونوس اضافی اگر p>=0.55 و FQ>=45
    bonus = 0.0
    if p >= 0.55 and feature_quality >= 45:
        bonus = 8.0
    if p >= 0.60 and feature_quality >= 55:
        bonus = 15.0

    return float(blended * harmony + bonus)


def _opt_evaluate_config_with_rows(loaded_datasets, params, spread_raw):
    """Like _opt_evaluate_config but also returns feature rows for FQ."""
    total_signals, total_wins = 0, 0
    signal_rows = []
    for item in loaded_datasets:
        df, tick_manager = item['df'], item['tm']
        signals, _funnel = build_pivot_settlement_signals(
            df, params['swing_n'], params['ab_min_bars'], params['ab_max_bars'], params['ab_atr_mult_min'],
            params['ab_extended_min'], params['ab_extended_max'], params['bc_min_bars'],
            params['bc_retrace_min'], params['bc_retrace_max'], params['box_scale'],
            params['signal_search_bars'], params['signal_atr_mult'], params['signal_body_ratio_min'],
            params['signal_wick_reject_ratio'], params['allow_fl_candle'],
            params['confirm_search_bars'], params['confirm_atr_mult'], params['sl_buffer_atr_mult'],
            params['cd_ab_mode'], params['ab_noise_fraction_max'], params['ab2_discount_mult']
        )
        if params['use_trend_gate'] and signals and 'EMA_Stack_Score' in df.columns:
            filtered=[]
            for s in signals:
                try: trend_val = df['EMA_Stack_Score'].loc[s['B_time']]
                except Exception: continue
                if pd.isna(trend_val): continue
                prevailing_up = trend_val > 0
                if params['trend_gate_mode'] == "خلاف روند (Counter-Trend Reversal)":
                    keep = (prevailing_up and s['dir'] == -1) or ((not prevailing_up) and s['dir'] == 1)
                else:
                    keep = (prevailing_up and s['dir'] == 1) or ((not prevailing_up) and s['dir'] == -1)
                if keep: filtered.append(s)
            signals = filtered
        for sig in signals:
            entry_price = sig['entry_price']; risk = sig['risk_raw']
            tp = entry_price - params['rr'] * risk if sig['dir'] == -1 else entry_price + params['rr'] * risk
            try: entry_loc = df.index.get_loc(sig['time'])
            except Exception: continue
            limit_idx = min(entry_loc + int(params['max_bars']), len(df) - 1)
            future = tick_manager.get_ticks_series(sig['time'], df.index[limit_idx])
            if len(future) > 0:
                status, _ = simulate_race_ps(future.values, sig['dir'], sig['M1_SL'], tp, spread_raw)
                is_winner = 1 if status == 1 else 0
            else:
                is_winner = 0
            total_signals += 1; total_wins += is_winner
            try:
                row = df.iloc[entry_loc].to_dict(); row['Target_Class']=is_winner
                for mk in ['PS_AB_Range_ATR','PS_AB_Bars','PS_BC_Retrace_Ratio','PS_BC_Bars','PS_CD_AB_Ratio','PS_Bars_Signal_To_Trigger','PS_FL_Candle_Used']:
                    if mk in sig: row[mk]=sig[mk]
                signal_rows.append(row)
            except Exception: pass
    return total_signals, total_wins, signal_rows

def run_pivot_optimization_ga(datasets, min_samples_target, population_size, n_generations,
                               elite_frac, mutation_rate, immigrant_frac, spread_raw, seed,
                               progress=gr.Progress()):
    if not datasets or "No Data Found" in datasets:
        return "❌ دیتاست یافت نشد.", None, None, None, None, None

    loaded_datasets = _opt_preload_datasets(datasets)
    if not loaded_datasets:
        return "❌ فایل فیچرهای دیتاست‌های انتخابی یافت نشد.", None, None, None, None, None

    rng = np.random.default_rng(int(seed) if seed else None)
    population_size = max(4, int(population_size))
    n_generations = max(1, int(n_generations))
    n_elite = max(1, int(round(population_size * float(elite_frac))))
    n_immigrants = max(0, int(round(population_size * float(immigrant_frac))))

    population = [_opt_sample_params(rng) for _ in range(population_size)]

    all_time_best_row = None
    all_time_best_fitness = -float('inf')
    leaderboard_rows = []
    generation_history = []

    total_steps = n_generations * population_size
    step = 0

    for gen in range(n_generations):
        evaluated = []
        for ind in population:
            total_signals, wins, _rows = _opt_evaluate_config_with_rows(loaded_datasets, ind, float(spread_raw))
            fq = _fast_feature_quality_score(_rows) if total_signals >= 25 else 0.0
            fit = _opt_fitness(total_signals, wins, int(min_samples_target), feature_quality=fq, require_wins_gt_losses=True, min_winrate=0.50)
            step += 1
            progress(step / total_steps, desc=f"🧬 نسل {gen + 1}/{n_generations} | فرد {len(evaluated) + 1}/{population_size}")

            losses = total_signals - wins
            class1_pct = (wins / total_signals * 100) if total_signals > 0 else 0.0

            row = dict(ind)
            row['spread_raw'] = float(spread_raw)
            row.update({
                'Generation': gen + 1,
                'Total_Signals': total_signals,
                'Wins_Class1': wins,
                'Losses_Class0': losses,
                'Win_Edge': int(wins - losses),
                'Class1_%': round(class1_pct, 2),
                'Wins_GT_Losses': bool(wins > losses),
                'Meets_Min_Samples': total_signals >= int(min_samples_target),
                'Feature_Quality': round(fq, 2),
                'Fitness': round(fit, 2),
            })

            evaluated.append((fit, ind))
            leaderboard_rows.append(row)

            if fit > all_time_best_fitness:
                all_time_best_fitness = fit
                all_time_best_row = row

        evaluated.sort(key=lambda x: x[0], reverse=True)
        gen_fitnesses = [e[0] for e in evaluated]
        gen_rows = [r for r in leaderboard_rows if r['Generation'] == gen + 1]
        gen_rows.sort(key=lambda r: r['Fitness'], reverse=True)
        generation_history.append({
            'Generation': gen + 1,
            'Best_Fitness': round(gen_fitnesses[0], 2),
            'Avg_Fitness': round(float(np.mean(gen_fitnesses)), 2),
            'Best_Total_Signals': gen_rows[0]['Total_Signals'],
            'Best_Wins': gen_rows[0]['Wins_Class1'],
            'Best_Losses': gen_rows[0]['Losses_Class0'],
            'Best_Win_Edge': gen_rows[0].get('Win_Edge', gen_rows[0]['Wins_Class1'] - gen_rows[0]['Losses_Class0']),
            'Best_Class1_%': gen_rows[0]['Class1_%'],
            'Best_Feature_Quality': gen_rows[0].get('Feature_Quality', 0),
            'Best_Wins_GT_Losses': gen_rows[0].get('Wins_GT_Losses', False),
        })

        if gen == n_generations - 1:
            break  # نسل آخر نیازی به تولید فرزند ندارد

        # --- ساخت نسل بعدی: Elitism + Crossover/Mutation + مهاجرِ تصادفی ---
        elites = [ind for _fit, ind in evaluated[:n_elite]]
        new_population = list(elites)

        while len(new_population) < (population_size - n_immigrants):
            parent_a = _opt_tournament_select(evaluated, k=3, rng=rng)
            parent_b = _opt_tournament_select(evaluated, k=3, rng=rng)
            child = _opt_crossover(parent_a, parent_b, rng)
            child = _opt_mutate(child, rng, float(mutation_rate))
            new_population.append(child)

        while len(new_population) < population_size:
            new_population.append(_opt_sample_params(rng))  # مهاجرِ تصادفی برای حفظ تنوع ژنتیکی

        population = new_population

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values('Fitness', ascending=False).reset_index(drop=True)
    history_df = pd.DataFrame(generation_history)

    # --- نمودار همگرایی (شبیه نمودار Optimization Result در متاتریدر) ---
    convergence_path = os.path.join(DATA_DIR, "HIPO_PivotSettlement_GA_Convergence.png")
    try:
        fig, ax1 = plt.subplots(figsize=(11, 5), facecolor='#0b0f19')
        ax1.set_facecolor('#0b0f19')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values(): spine.set_color('#333')
        ax1.plot(history_df['Generation'], history_df['Best_Fitness'], color='#00ffcc', linewidth=2, marker='o', label='بهترین فیتنس نسل')
        ax1.plot(history_df['Generation'], history_df['Avg_Fitness'], color='#7000ff', linewidth=1.5, linestyle='--', label='میانگین فیتنس نسل')
        ax1.set_xlabel('نسل (Generation)', color='white')
        ax1.set_ylabel('فیتنس (Wins>Losses + FeatureQuality)', color='white')
        ax1.set_title('همگرایی GA v5 — Wins>Losses + Feature Quality', color='#00f2ff', fontweight='bold')
        ax1.legend(facecolor='#161b22', labelcolor='white')
        ax1.grid(alpha=0.15)
        plt.tight_layout()
        fig.savefig(convergence_path, dpi=120)
        plt.close(fig)
    except Exception:
        plt.close('all')
        convergence_path = None

    # اولویت: min_samples + Wins>Losses
    qualifying = leaderboard_df[
        (leaderboard_df['Meets_Min_Samples'] == True) & (leaderboard_df.get('Wins_GT_Losses', False) == True)
    ] if 'Wins_GT_Losses' in leaderboard_df.columns else leaderboard_df[leaderboard_df['Meets_Min_Samples'] == True]
    if 'Wins_GT_Losses' in leaderboard_df.columns:
        q_edge = leaderboard_df[(leaderboard_df['Meets_Min_Samples'] == True) & (leaderboard_df['Wins_GT_Losses'] == True)]
        q_sample = leaderboard_df[leaderboard_df['Meets_Min_Samples'] == True]
    else:
        q_edge = leaderboard_df.iloc[0:0]
        q_sample = leaderboard_df[leaderboard_df['Meets_Min_Samples'] == True]

    if len(q_edge) > 0:
        best_row = q_edge.sort_values('Fitness', ascending=False).iloc[0].to_dict()
        summary = (
            f"🏆 بهترین فرد GA v5 (نسل {int(best_row['Generation'])} | "
            f"{n_generations}×{population_size}={total_steps} ارزیابی)\n\n"
            f"   ✅ Wins > Losses  برقرار است\n"
            f"   • Total Signals: **{int(best_row['Total_Signals'])}**\n"
            f"   • Wins (Class1): **{int(best_row['Wins_Class1'])}**\n"
            f"   • Losses (Class0): **{int(best_row['Losses_Class0'])}**\n"
            f"   • Win Edge (W-L): **{int(best_row.get('Win_Edge', best_row['Wins_Class1']-best_row['Losses_Class0']))}**\n"
            f"   • WinRate: **{best_row['Class1_%']}%**\n"
            f"   • Feature Quality: **{best_row.get('Feature_Quality', 0)}/100**\n"
            f"   • Fitness: **{best_row['Fitness']}**\n\n"
            f"بعد از Apply، تب «گزارش کیفیت فیچر» را اجرا کنید."
        )
    elif len(q_sample) > 0:
        best_row = q_sample.sort_values('Fitness', ascending=False).iloc[0].to_dict()
        summary = (
            f"⚠️ به min_samples رسیدیم ولی **Wins > Losses** در بهترین‌ها برقرار نشد.\n"
            f"   Total={int(best_row['Total_Signals'])} | W={int(best_row['Wins_Class1'])} | "
            f"L={int(best_row['Losses_Class0'])} | WR={best_row['Class1_%']}% | "
            f"FQ={best_row.get('Feature_Quality', 0)}\n"
            f"➡️ RR را کم کنید (1.0)، نسل/جمعیت را زیاد کنید، یا spread را واقعی بگذارید."
        )
    else:
        best_row = leaderboard_df.sort_values('Fitness', ascending=False).iloc[0].to_dict()
        summary = (
            f"⚠️ در {n_generations} نسل به حداقل {int(min_samples_target)} نمونه نرسیدیم.\n"
            f"نزدیک‌ترین: n={int(best_row['Total_Signals'])} WR={best_row['Class1_%']}% "
            f"FQ={best_row.get('Feature_Quality', 0)}\n"
            f"➡️ min_samples را کم کنید یا قیف را شل‌تر کنید."
        )

    report_path = os.path.join(DATA_DIR, "HIPO_PivotSettlement_GA_Optimization_Report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            'best_config': best_row,
            'generation_history': generation_history,
            'full_leaderboard': leaderboard_rows,
            'min_samples_target': int(min_samples_target),
            'population_size': population_size,
            'n_generations': n_generations,
            'datasets': datasets
        }, f, ensure_ascii=False, indent=2, default=str)

    display_cols = ['Generation', 'Total_Signals', 'Wins_Class1', 'Losses_Class0', 'Win_Edge', 'Class1_%', 'Wins_GT_Losses', 'Feature_Quality', 'Meets_Min_Samples', 'Fitness']
    other_cols = [c for c in leaderboard_df.columns if c not in display_cols]
    top_table = leaderboard_df[display_cols + other_cols].head(30)

    return summary, top_table, history_df, convergence_path, best_row, report_path


def apply_best_optimization_config(best_config_state, datasets, progress=gr.Progress()):
    """
    پیکربندیِ برگزیده‌ی الگوریتم ژنتیک را می‌گیرد و *بدون هیچ تغییری در خودِ منطق*،
    آن را با فراخوانیِ process_labeling_pivot_settlement (همان تابع دست‌نخورده‌ی
    بخش دستی) اجرا می‌کند تا پارکت لیبل‌شده، تصاویر، آمار و قیف تشخیصی — دقیقاً
    مثل حالت دستی — تولید شوند.
    """
    if not best_config_state:
        return "❌ ابتدا از سربرگ بالا اپتیمایزیشن را اجرا کنید.", None, None, None, None

    bc = best_config_state
    return process_labeling_pivot_settlement(
        datasets, bc['swing_n'], bc['ab_min_bars'], bc['ab_max_bars'], bc['ab_atr_mult_min'],
        bc['ab_extended_min'], bc['ab_extended_max'], bc['bc_min_bars'],
        bc['bc_retrace_min'], bc['bc_retrace_max'], bc['box_scale'],
        bc['signal_search_bars'], bc['signal_atr_mult'], bc['signal_body_ratio_min'],
        bc['signal_wick_reject_ratio'], bc['allow_fl_candle'],
        bc['confirm_search_bars'], bc['confirm_atr_mult'], bc['sl_buffer_atr_mult'],
        bc['cd_ab_mode'], bc['rr'], float(bc.get('spread_raw', 0.0002)), bc['max_bars'],
        bc['use_trend_gate'], bc['trend_gate_mode'],
        bc['ab_noise_fraction_max'], bc['ab2_discount_mult'],
        progress=progress
    )




# =============================================================================
# 8C. گزارش کیفیت فیچر ↔ لیبل (بعد از لیبل‌گذاری)
# =============================================================================
_FQ_DROP = {
    "Open", "High", "Low", "Close", "Volume", "Tick_Up_Count", "Tick_Down_Count",
    "Bid", "Ask", "Target_Class", "Signal_Dir", "entry_price", "M1_SL", "M1_TP",
    "max_bars", "pair", "Raw_ATR", "index", "level_0", "Time",
}

def _fq_is_num(s):
    try:
        return bool(pd.api.types.is_numeric_dtype(s))
    except Exception:
        return False

def run_feature_label_quality_report(datasets, max_features=35, min_quality=25, progress=gr.Progress()):
    """
    گزارش کامل کیفیت فیچرها نسبت به لیبل‌های ساخته‌شده.
    خروجی: selected_features.json + HTML + جدول رتبه‌بندی
    """
    if not datasets or "No Data Found" in datasets:
        return "❌ دیتاست لیبل‌شده یافت نشد. اول Manual یا Apply GA را بزنید.", None, None, None, None

    progress(0.05, desc="Loading labeled...")
    frames = []
    for name in datasets:
        # accept both bare name and with _Labeled
        base = name.replace("_Labeled.parquet", "").replace("_Labeled", "")
        path = os.path.join(DATA_DIR, f"{base}_Labeled.parquet")
        # also if user selected filename
        alt = os.path.join(DATA_DIR, name if name.endswith(".parquet") else "")
        if os.path.exists(path):
            d = pd.read_parquet(path)
            d["__dataset__"] = base
            frames.append(d)
        elif alt and os.path.exists(alt):
            d = pd.read_parquet(alt)
            d["__dataset__"] = base
            frames.append(d)
    if not frames:
        # try glob all labeled
        labs = glob.glob(os.path.join(DATA_DIR, "*_Labeled.parquet"))
        for path in labs:
            d = pd.read_parquet(path)
            d["__dataset__"] = os.path.basename(path).replace("_Labeled.parquet", "")
            frames.append(d)
    if not frames:
        return "❌ هیچ *_Labeled.parquet پیدا نشد.", None, None, None, None

    df = pd.concat(frames, axis=0).sort_index()
    if "Target_Class" not in df.columns:
        return "❌ ستون Target_Class نیست.", None, None, None, None

    n = len(df)
    wins = int((df["Target_Class"] == 1).sum())
    losses = n - wins
    wr = wins / max(1, n)
    progress(0.2, desc="Scoring features...")

    y = df["Target_Class"].astype(int).values
    cols = [c for c in df.columns if c not in _FQ_DROP and not str(c).startswith("__") and _fq_is_num(df[c])]
    if not cols:
        return "❌ فیچر عددی نیست.", None, None, None, None
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    Xm = X.fillna(X.median(numeric_only=True))

    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(Xm.values.astype(np.float64), y, random_state=42,
                                 n_neighbors=min(5, max(1, n // 20)))
    except Exception:
        mi = np.zeros(len(cols))

    rows = []
    for i, c in enumerate(cols):
        v = Xm[c].values.astype(np.float64)
        if np.std(v) < 1e-12:
            auc, r = 0.5, 0.0
        else:
            order = np.argsort(v)
            yo = y[order]
            n_pos = float(yo.sum()); n_neg = float(len(yo) - n_pos)
            if n_pos < 1 or n_neg < 1:
                auc = 0.5
            else:
                ranks = np.arange(1, len(yo) + 1, dtype=np.float64)
                auc = (ranks[yo == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                auc = max(float(auc), 1 - float(auc))
            r = float(np.corrcoef(v, y)[0, 1])
            if np.isnan(r):
                r = 0.0
        q = (0.45 * float(np.clip((auc - 0.5) / 0.15, 0, 1)) * 100
             + 0.35 * float(np.clip(mi[i] / 0.12, 0, 1)) * 100
             + 0.20 * float(np.clip(abs(r) / 0.25, 0, 1)) * 100)
        verdict = "KEEP_STRONG" if q >= 55 else ("KEEP" if q >= 35 else ("WEAK" if q >= 20 else "DROP"))
        rows.append(dict(feature=c, AUC_abs=round(auc, 4), MI=round(float(mi[i]), 5),
                         abs_corr=round(abs(r), 4), quality_0_100=round(q, 2), verdict=verdict))
    feat_df = pd.DataFrame(rows).sort_values("quality_0_100", ascending=False).reset_index(drop=True)

    # select + dedup
    progress(0.55, desc="Selecting features...")
    selected = []
    cands = feat_df[feat_df["quality_0_100"] >= float(min_quality)]
    if len(cands) == 0:
        cands = feat_df.head(int(max_features))
    for _, row in cands.iterrows():
        c = row["feature"]
        if c not in X.columns or len(selected) >= int(max_features):
            if len(selected) >= int(max_features):
                break
            continue
        v = X[c].fillna(X[c].median()).values
        ok = True
        for s in selected:
            vs = X[s].fillna(X[s].median()).values
            if np.std(v) < 1e-12 or np.std(vs) < 1e-12:
                continue
            rr = abs(np.corrcoef(v, vs)[0, 1])
            if not np.isnan(rr) and rr >= 0.92:
                ok = False
                break
        if ok:
            selected.append(c)
    for _, row in feat_df.iterrows():
        if str(row["feature"]).startswith(("PSA_", "LSW_", "FVG_", "PS_")) and row["verdict"] in ("KEEP", "KEEP_STRONG"):
            if row["feature"] not in selected and len(selected) < int(max_features) + 8:
                selected.append(row["feature"])

    # multivariate probe + shuffle
    progress(0.7, desc="Multivariate probe + shuffle...")
    probe = {"cv_pr_auc": 0.0, "lift": 0.0, "shuffle_p": 1.0, "real": False, "base": round(wr, 4)}
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score
        if len(selected) >= 5 and n >= 30:
            Xs = X[selected].fillna(X[selected].median()).astype(np.float32)
            tscv = TimeSeriesSplit(n_splits=min(4, max(2, n // 25)), gap=5)
            sc = []
            for tr, te in tscv.split(Xs):
                if y[tr].sum() < 3 or y[te].sum() < 2:
                    continue
                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.08,
                                                     min_samples_leaf=max(5, len(tr)//40),
                                                     l2_regularization=1.0, random_state=42)
                clf.fit(Xs.iloc[tr], y[tr])
                sc.append(average_precision_score(y[te], clf.predict_proba(Xs.iloc[te])[:, 1]))
            if sc:
                real = float(np.mean(sc))
                rng = np.random.default_rng(42)
                sh = []
                for _ in range(10):
                    ys = y.copy(); rng.shuffle(ys)
                    fs = []
                    for tr, te in tscv.split(Xs):
                        if ys[tr].sum() < 3 or ys[te].sum() < 2:
                            continue
                        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=50, learning_rate=0.08,
                                                             min_samples_leaf=max(5, len(tr)//40),
                                                             l2_regularization=1.0, random_state=0)
                        clf.fit(Xs.iloc[tr], ys[tr])
                        fs.append(average_precision_score(ys[te], clf.predict_proba(Xs.iloc[te])[:, 1]))
                    if fs:
                        sh.append(float(np.mean(fs)))
                pval = (1 + sum(1 for s in sh if s >= real - 1e-12)) / (len(sh) + 1) if sh else 1.0
                lift = real / (wr + 1e-9)
                probe = dict(cv_pr_auc=round(real, 4), cv_std=round(float(np.std(sc)), 4),
                             base=round(wr, 4), lift=round(lift, 3), shuffle_p=round(float(pval), 4),
                             real=bool(pval < 0.15 and lift > 1.08))
    except Exception as e:
        probe["error"] = str(e)

    # global score
    top10 = feat_df.head(10)
    uni = float(np.clip(top10["quality_0_100"].mean() / 70, 0, 1) * 35)
    multi = 30 * float(np.clip((probe.get("lift", 1) - 1) / 0.5, 0, 1))
    if not probe.get("real"):
        multi *= 0.5
    sample_s = 10 * float(np.clip(np.log1p(n) / np.log1p(2000), 0, 1))
    # win edge component: reward wins>losses
    edge_s = 10.0 if wins > losses else (5.0 * wr / 0.5 if wr > 0 else 0.0)
    edge_s = float(np.clip(edge_s, 0, 15))
    strong = int((feat_df["verdict"] == "KEEP_STRONG").sum())
    cover = 10 * float(np.clip(strong / 10, 0, 1))
    gscore = round(float(np.clip(uni + multi + sample_s + edge_s + cover, 0, 100)), 2)

    if gscore >= 65 and probe.get("real") and wins > losses:
        interp = "✅ فیچرها با لیبل هم‌راستا هستند، Wins>Losses برقرار است، سیگنال آماری واقعی است. برو Training."
    elif gscore >= 45 and wins > losses:
        interp = "⚠️ کیفیت متوسط رو به خوب. Wins>Losses OK است ولی هنوز برای 60% پایدار نمونه/OOS بیشتر لازم است."
    elif wins <= losses:
        interp = "❌ بردها ≤ باخت‌ها. اول GA را طوری تنظیم کنید که Wins>Losses شود (RR پایین‌تر، قیف بهتر)."
    else:
        interp = "❌ فیچر↔لیبل ضعیف. Feature Engine v25 (PSA/LSW/FVG) را دوباره بسازید و GA را تکرار کنید."

    # save artifacts
    progress(0.9, desc="Saving reports...")
    sel_path = os.path.join(DATA_DIR, "selected_features.json")
    payload = {
        "selected_features": selected,
        "global_score": gscore,
        "probe": probe,
        "n_samples": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr * 100, 2),
        "wins_gt_losses": bool(wins > losses),
        "interpretation": interp,
        "top_features": feat_df.head(40).to_dict(orient="records"),
    }
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    html_path = os.path.join(DATA_DIR, "HIPO_FeatureLabel_Quality_Report.html")
    color = "#00ff88" if gscore >= 60 else ("#ffcc00" if gscore >= 40 else "#ff3366")
    edge_badge = "✅ Wins > Losses" if wins > losses else "❌ Wins ≤ Losses"
    rows_html = "".join(
        f"<tr><td>{r.feature}</td><td>{r.AUC_abs}</td><td>{r.MI}</td><td>{r.quality_0_100}</td><td>{r.verdict}</td></tr>"
        for _, r in feat_df.head(40).iterrows()
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html lang=fa><head><meta charset=utf-8><title>Feature↔Label Quality</title>
<style>body{{background:#0b0f19;color:#e6edf3;font-family:Tahoma,monospace;padding:24px;direction:rtl}}
h1{{color:#00f2ff}}.score{{font-size:64px;color:{color};font-weight:bold}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #30363d;padding:6px;text-align:center}}
th{{background:#21262d;color:#00f2ff}}.badge{{display:inline-block;padding:6px 12px;margin:4px;background:#21262d;border-radius:8px}}</style></head>
<body><h1>🔬 گزارش کیفیت فیچر ↔ لیبل</h1>
<div class=score>{gscore}</div>
<div class=badge>n={n}</div><div class=badge>W={wins}</div><div class=badge>L={losses}</div>
<div class=badge>WR={wr*100:.1f}%</div><div class=badge>{edge_badge}</div>
<div class=badge>selected={len(selected)}</div>
<div class=badge>CV PR-AUC={probe.get('cv_pr_auc')}</div>
<div class=badge>Lift={probe.get('lift')}</div>
<div class=badge>Shuffle p={probe.get('shuffle_p')}</div>
<p><b>{interp}</b></p>
<h2>Top 40 Features</h2>
<table><tr><th>Feature</th><th>AUC</th><th>MI</th><th>Q</th><th>Verdict</th></tr>{rows_html}</table>
<h2>Selected</h2><pre>{json.dumps(selected, ensure_ascii=False, indent=2)}</pre>
</body></html>""")

    # chart
    fig_path = os.path.join(DATA_DIR, "HIPO_FeatureQuality_Top.png")
    try:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0b0f19")
        ax.set_facecolor("#0b0f19")
        top = feat_df.head(15)
        ax.barh(top["feature"][::-1], top["quality_0_100"][::-1], color="#00f2ff")
        ax.tick_params(colors="white")
        ax.axvline(35, color="#ffaa00", ls="--", alpha=0.7)
        ax.axvline(55, color="#00ff88", ls="--", alpha=0.7)
        ax.set_title("Top Feature Quality vs Label", color="#00ff88", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#333")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
    except Exception:
        plt.close("all")
        fig_path = None

    msg = (
        f"### 🔬 گزارش کیفیت فیچر ↔ لیبل\\n\\n"
        f"**Global Score: `{gscore}/100`**\\n\\n"
        f"- نمونه‌ها: **{n}** | برد: **{wins}** | باخت: **{losses}** | "
        f"{'✅ Wins>Losses' if wins > losses else '❌ Wins≤Losses'}\\n"
        f"- WinRate: **{wr*100:.1f}%** | Selected: **{len(selected)}** "
        f"(KEEP_STRONG={strong})\\n"
        f"- CV PR-AUC: **{probe.get('cv_pr_auc')}** | Lift: **{probe.get('lift')}×** | "
        f"Shuffle p: **{probe.get('shuffle_p')}** "
        f"→ {'✅ REAL' if probe.get('real') else '⚠️ weak/noise'}\\n\\n"
        f"**تفسیر:** {interp}\\n\\n"
        f"📁 `selected_features.json` و گزارش HTML ذخیره شد."
    )
    progress(1.0, desc="Done")
    return msg, feat_df.head(40), fig_path, html_path, sel_path


def get_labeled_datasets_for_quality():
    files = glob.glob(os.path.join(DATA_DIR, "*_Labeled.parquet"))
    names = sorted({os.path.basename(f).replace("_Labeled.parquet", "") for f in files})
    return names if names else ["No Data Found"]


# =============================================================================
# 9. رابط کاربری (Gradio UI - Pivot Settlement Edition)
# =============================================================================

with gr.Blocks(title="HIPO LABELING FORGE v5 COMPLETE") as web_app:
    gr.HTML("""
        <div style="text-align:center;border-bottom:1px solid #333;padding-bottom:16px;margin-bottom:16px">
            <h1 style="color:#00f2ff;font-family:monospace;font-size:28px;margin:0">
                🧭 HIPO PIVOT SETTLEMENT FORGE v5.0 COMPLETE
            </h1>
            <p style="color:#00ff88;font-family:monospace;margin-top:8px">
                Manual Labeling · GA (Wins&gt;Losses + FeatureQuality) · Feature↔Label Quality Report
            </p>
        </div>
    """)

    with gr.Tabs():
        # ===================== TAB 1 MANUAL =====================
        with gr.Tab("🎯 استخراج دستی"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ موج AB / BC")
                    w_data = gr.CheckboxGroup(
                        choices=get_available_datasets(),
                        label="دیتاست‌های Features",
                        value=[get_available_datasets()[0]] if get_available_datasets() else [],
                    )
                    w_swing_n = gr.Slider(1, 10, value=1, step=1, label="Swing N (پیشنهاد تست‌شده: 1)")
                    w_ab_min_bars = gr.Slider(2, 10, value=2, step=1, label="حداقل کندل AB")
                    w_ab_max_bars = gr.Slider(3, 20, value=12, step=1, label="حداکثر کندل AB")
                    w_ab_atr_mult = gr.Slider(0.5, 5.0, value=1.1, step=0.1, label="حداقل AB/ATR")
                    w_ab_ext_min = gr.Slider(1.0, 6.0, value=1.5, step=0.1, label="حد پایین AB بلند (ATR)")
                    w_ab_ext_max = gr.Slider(2.0, 10.0, value=6.0, step=0.1, label="حد بالای AB بلند (ATR)")
                    w_ab_noise_fraction = gr.Slider(0.1, 0.6, value=0.50, step=0.02, label="نسبت نویز مجاز AB")
                    w_ab2_discount = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="تخفیف ATR برای AB دوکندلی")
                    w_cd_ab_mode = gr.Dropdown(
                        choices=["CD > AB (طبق نقاط تکرارشده در PDF)", "AB > CD (طبق خط اول PDF)"],
                        value="CD > AB (طبق نقاط تکرارشده در PDF)",
                        label="قانون AB/CD",
                    )
                with gr.Column():
                    gr.Markdown("### 🧲 BC / باکس / سیگنال")
                    w_bc_min_bars = gr.Slider(1, 15, value=2, step=1, label="حداقل کندل BC")
                    w_bc_retrace_min = gr.Slider(0.05, 0.6, value=0.15, step=0.01, label="حداقل اصلاح BC")
                    w_bc_retrace_max = gr.Slider(0.2, 0.8, value=0.65, step=0.01, label="حداکثر اصلاح BC")
                    w_box_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="ضریب باکس نقدینگی")
                    w_signal_search_bars = gr.Slider(5, 150, value=100, step=5, label="مهلت جستجوی سیگنال")
                    w_signal_atr_mult = gr.Slider(0.1, 2.0, value=0.4, step=0.05, label="رنج سیگنال/ATR")
                    w_signal_body_ratio = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="نسبت بدنه سیگنال")
                    w_signal_wick_reject = gr.Slider(0.3, 3.0, value=1.5, step=0.05, label="حداکثر شدوی مخالف")
                    w_allow_fl = gr.Checkbox(value=True, label="فعال‌سازی FL candle")
                    w_confirm_search_bars = gr.Slider(1, 50, value=20, step=1, label="مهلت کندل تایید")
                    w_confirm_atr_mult = gr.Slider(0.1, 2.0, value=0.3, step=0.05, label="رنج تایید/ATR")
                with gr.Column():
                    gr.Markdown("### 🏁 ریسک و داوری")
                    w_sl_buffer = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="بافر SL (ATR)")
                    w_rr = gr.Slider(1.0, 6.0, value=1.0, step=0.1, label="R:R (برای Wins>Losses: 1.0–1.5)")
                    w_spread = gr.Number(value=0.00012, label="اسپرد (EUR≈0.00012 | XAU=0.35)")
                    w_max_bars = gr.Slider(10, 300, value=60, step=5, label="Max Bars")
                    w_trend_gate = gr.Checkbox(value=False, label="فیلتر روند EMA_Stack")
                    w_trend_gate_mode = gr.Dropdown(
                        choices=["خلاف روند (Counter-Trend Reversal)", "هم‌جهت روند (With-Trend Continuation)"],
                        value="خلاف روند (Counter-Trend Reversal)",
                        label="حالت فیلتر روند",
                    )
            w_btn = gr.Button("🔥 EXTRACT PIVOT SETTLEMENT SIGNALS", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=2):
                    w_msg = gr.Textbox(label="وضعیت", lines=6)
                    w_tail = gr.DataFrame(label="نمونه سیگنال‌ها")
                with gr.Column(scale=1):
                    w_stats = gr.DataFrame(label="توزیع Win/Loss")
                    w_zip = gr.File(label="تصاویر ZIP")
            w_funnel = gr.DataFrame(label="قیف رد سیگنال")
            w_btn.click(
                process_labeling_pivot_settlement,
                inputs=[w_data, w_swing_n, w_ab_min_bars, w_ab_max_bars, w_ab_atr_mult, w_ab_ext_min, w_ab_ext_max,
                        w_bc_min_bars, w_bc_retrace_min, w_bc_retrace_max, w_box_scale,
                        w_signal_search_bars, w_signal_atr_mult, w_signal_body_ratio, w_signal_wick_reject, w_allow_fl,
                        w_confirm_search_bars, w_confirm_atr_mult, w_sl_buffer, w_cd_ab_mode,
                        w_rr, w_spread, w_max_bars, w_trend_gate, w_trend_gate_mode,
                        w_ab_noise_fraction, w_ab2_discount],
                outputs=[w_msg, w_tail, w_stats, w_zip, w_funnel],
            )

        # ===================== TAB 2 GA =====================
        with gr.Tab("🧬 اپتیمایز ژنتیک (Wins>Losses + FeatureQuality)"):
            gr.Markdown(
                """
### هدف GA v5
1. **Wins_Class1 > Losses_Class0** (قید اصلی — بردها بیشتر از باخت‌ها)
2. **Feature_Quality** بالا (فیچرها Win/Loss را جدا کنند)
3. **تعداد نمونه** کافی (min_samples)

منطق کشف سیگنال عوض نمی‌شود؛ فقط پارامترها تکامل می‌یابند.
                """
            )
            with gr.Row():
                w_opt_data = gr.CheckboxGroup(
                    choices=get_available_datasets(),
                    label="دیتاست‌ها",
                    value=[get_available_datasets()[0]] if get_available_datasets() else [],
                )
                w_opt_min_samples = gr.Slider(
                    30, 5000, value=80, step=10,
                    label="حداقل نمونه (قید سخت)",
                    info="روی 2 سال M15 معمولاً 80–300 واقع‌بینانه است",
                )
            with gr.Row():
                w_opt_population = gr.Slider(8, 80, value=20, step=2, label="جمعیت")
                w_opt_generations = gr.Slider(3, 40, value=12, step=1, label="نسل‌ها")
                w_opt_elite = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Elitism")
            with gr.Row():
                w_opt_mutation = gr.Slider(0.05, 0.6, value=0.25, step=0.05, label="Mutation")
                w_opt_immigrant = gr.Slider(0.0, 0.4, value=0.20, step=0.05, label="Immigrants (تنوع)")
                w_opt_spread = gr.Number(value=0.00012, label="اسپرد")
                w_opt_seed = gr.Number(value=42, label="Seed")

            w_opt_btn = gr.Button("🚀 شروع GA v5 (Wins>Losses + FQ)", variant="primary", size="lg")
            w_opt_summary = gr.Markdown()
            with gr.Row():
                w_opt_convergence = gr.Image(label="همگرایی", type="filepath")
                w_opt_history = gr.DataFrame(label="تاریخچه نسل")
            w_opt_table = gr.DataFrame(label="۳۰ فرد برتر")
            w_opt_report = gr.File(label="گزارش JSON")
            state_best_config = gr.State(None)

            w_opt_btn.click(
                run_pivot_optimization_ga,
                inputs=[w_opt_data, w_opt_min_samples, w_opt_population, w_opt_generations,
                        w_opt_elite, w_opt_mutation, w_opt_immigrant, w_opt_spread, w_opt_seed],
                outputs=[w_opt_summary, w_opt_table, w_opt_history, w_opt_convergence, state_best_config, w_opt_report],
            )

            gr.Markdown("### اعمال بهترین تنظیمات → ساخت Labeled parquet")
            w_opt_apply_btn = gr.Button("📦 Apply Best Config & Build Labels", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    w_opt_final_msg = gr.Textbox(label="وضعیت", lines=6)
                    w_opt_final_tail = gr.DataFrame(label="سیگنال‌ها")
                with gr.Column(scale=1):
                    w_opt_final_stats = gr.DataFrame(label="Win/Loss")
                    w_opt_final_zip = gr.File(label="ZIP تصاویر")
            w_opt_final_funnel = gr.DataFrame(label="قیف")
            w_opt_apply_btn.click(
                apply_best_optimization_config,
                inputs=[state_best_config, w_opt_data],
                outputs=[w_opt_final_msg, w_opt_final_tail, w_opt_final_stats, w_opt_final_zip, w_opt_final_funnel],
            )

        # ===================== TAB 3 QUALITY =====================
        with gr.Tab("🔬 گزارش کیفیت فیچر ↔ لیبل"):
            gr.Markdown(
                """
### بعد از ساخت Labeled این تب را اجرا کنید
- رتبه‌بندی هر فیچر (AUC / MI / Quality)
- Global Score + Label-Shuffle
- ذخیره `selected_features.json` برای Training
- بررسی **Wins > Losses**
                """
            )
            with gr.Row():
                w_q_data = gr.CheckboxGroup(
                    choices=get_labeled_datasets_for_quality(),
                    label="دیتاست‌های Labeled",
                    value=[get_labeled_datasets_for_quality()[0]] if get_labeled_datasets_for_quality() else [],
                )
                w_q_max = gr.Slider(10, 80, value=35, step=1, label="حداکثر فیچر منتخب")
                w_q_min = gr.Slider(10, 60, value=25, step=1, label="حداقل Quality برای KEEP")
            w_q_btn = gr.Button("🚀 RUN FEATURE↔LABEL QUALITY REPORT", variant="primary", size="lg")
            w_q_msg = gr.Markdown()
            with gr.Row():
                w_q_table = gr.DataFrame(label="رتبه‌بندی فیچرها")
                w_q_fig = gr.Image(label="نمودار کیفیت", type="filepath")
            with gr.Row():
                w_q_html = gr.File(label="HTML Report")
                w_q_json = gr.File(label="selected_features.json")
            w_q_btn.click(
                run_feature_label_quality_report,
                inputs=[w_q_data, w_q_max, w_q_min],
                outputs=[w_q_msg, w_q_table, w_q_fig, w_q_html, w_q_json],
            )

web_app.queue().launch(share=True, inbrowser=True)
