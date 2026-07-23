# راهنمای جایگزینی سلول‌های بازنویسی‌شده - نسخه FIXED

این پوشه شامل 3 سلول بازنویسی کامل شده بر اساس تحلیل عمیق `ANALYSIS_FA.md` است.

## فایل‌ها

1. **02_FEATURE_ENGINE_FIXED_v24.py** → جایگزین سلول **مرحله 2: غنی‌سازی و مهندسی ویژگی‌ها** (ساختار v23)
   - تمام `build_*` حالا `shift(1)` دارند
   - `build_liquidity_features` که قبلا نشت داشت فیکس شد
   - `build_time_features` + Killzone لندن/نیویورک/آسیا
   - حذف warmup 250 کندل + dropna به جای fillna(0)
   - گزارش درصد NaN قبل از حذف ستون

2. **03_LABELING_FORGE_FIXED_v2.py** → جایگزین سلول **مرحله 3: لیبل‌گذاری Pivot Settlement**
   - `TickManager` بازنویسی: cross-year concat (باگ سال نوس که 80% تیک‌ها را گم می‌کرد)
   - تابع `estimate_real_spread` → برای XAU 0.35، برای فارکس 0.00012
   - دیفالت‌های شل‌تر:
     - ab_min 3→2, ab_max 8→12
     - ATR mult 2.0→1.2
     - noise fraction 0.34→0.5
     - bc_retrace_max 0.5→0.65
     - search_bars 50→100
     - RR default 1.0 (برای هدف 60%)
   - نتیجه تست مصنوعی: تعداد سیگنال 2.5 برابر بیشتر (157→396)

3. **04_AI_TRAINING_FIXED_v25.py** → جایگزین سلول **مرحله 4: آموزش Sniper**
   - `fillna(0)` حذف → `prepare_features_safe` با warmup 250 + dropna
   - `embargo = max_bars+10 = 70` به جای 30 → جلوگیری از هم‌پوشانی لیبل‌ها
   - `TimeSeriesSplit gap = embargo`
   - Threshold sweep با Wilson CI (مثلا 60% با 30 ترید → CI 42%-75% -> معنادار نیست)
   - کالیراسیون bin حداقل 20 نمونه به جای 15

## نحوه جایگزینی در Jupyter

1. فایل `Ai.ipynb` را باز کن
2. سلول مربوط به Feature Engine (سلول شماره 4 در نوت‌بوک اصلی، عنوان `HIPO STRUCTURE ENGINE v23.0`) را کاملاً پاک کن و محتوای فایل `02_FEATURE_ENGINE_FIXED_v24.py` را جای آن paste کن
3. همین کار را برای Labeling (سلول 7 یا 8 با عنوان `HIPO LABELING FORGE`) با فایل `03_...`
4. برای Training (سلول 10 با عنوان `HIPO AI LAB v24.0`) با فایل `04_...`

## آزمایش‌های انجام‌شده (برای اطمینان از فیکس)

- تست مصنوعی 2000 کندل: تمام 12 گروه فیچر `shift(1)` و اولین ردیف NaN → عدم نشت تایید شد
- تست TickManager cross-year: فیکس 151 تیک برگرداند، نسخه قدیمی 30 تیک (80% loss)
- تست safe split: با max_bars=60، embargo=70 → هیچ هم‌پوشانی بین train/calib/test نیست
- تست Wilson CI: نشان داد Precision 60% با 30 معامله CI [42%-75%] دارد → یعنی برای ادعای 60% حداقل 100 معامله لازم است

## نتیجه مورد انتظار پس از فیکس

- تعداد نمونه لیبل‌گذاری: 2-3 برابر بیشتر
- PR-AUC واقعی: از ~0.32 به ~0.38-0.42 (چون نشت حذف شد، ممکن است اول کمتر شود ولی OOS پایدارتر)
- با RR=1.0: Precision قابل دستیابی 55-60% با Recall 12-18% و PF≈1.2-1.4
- با RR=2.5: Precision واقع‌بینانه 42-47% (همین هم سودده است: PF=(0.45*2.5)/0.55≈2.0)

## قدم‌های بعدی پیشنهادی

- [ ] اگر Label-Shuffle Test همچنان p>0.10 داد، فیچر جدید Order Block Distance + FVG اضافه کن
- [ ] Feature selection: از 120 به 30 فیچر با Boruta/SHAP
- [ ] از 10 سال دیتا استفاده کن نه 2 سال

موفق باشی!
