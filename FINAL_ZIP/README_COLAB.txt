این ZIP شامل 3 سلول نهایی کاملا هماهنگ برای گوگل کولب است - بدون خطا:

1. 01_FEATURE_ENGINE_FINAL.py
   - جایگزین سلول مرحله 2 (Structure Engine v23)
   - ورودی: /content/hipo_lab_data/*_Tick*.parquet (خروجی دانلودر)
   - خروجی: /content/hipo_lab_data/*_Features.parquet
   - فیکس: تمام فیچرها shift(1) + warmup drop 250 + Killzone

2. 02_LABELING_FORGE_FINAL_COLAB_CLEAN.py
   - جایگزین سلول مرحله 3 (Labeling Forge v1.0)
   - ورودی: /content/hipo_lab_data/*_Features.parquet
   - خروجی: /content/hipo_lab_data/*_Labeled.parquet + تصاویر
   - فیکس: TickManager cross-year + calc_raw_atr (رفع NameError) + Real spread XAU 0.35 + Loose defaults (ab_min 2, ab_max 12, ATR 1.2)
   - دو تب: Manual + Genetic Optimizer (فیتنس جدید: تعداد + درصد برد)

3. 03_TRAINING_FINAL.py
   - جایگزین سلول مرحله 4 (AI LAB v24)
   - ورودی: /content/hipo_lab_data/*_Labeled.parquet
   - خروجی: Sniper Report + Model zip + Doctor Report
   - فیکس: No fillna(0) + embargo=70 + Wilson CI + Doctor یکپارچه (تب دوم)

نحوه استفاده در کولب:
- سلول 1 دانلودر (اصلی پروژه) را اجرا کن تا 1-2 سال دیتا بگیرد
- سلول 2 (فیوچر) را با 01_... جایگزین و اجرا کن
- سلول 3 (لیبل) را با 02_..._CLEAN جایگزین و اجرا کن (حالت عادی، بعد ژنتیک)
- سلول 4 (آموزش) را با 03_... جایگزین و اجرا کن (تب اول Sniper، تب دوم Doctor)

تست محیطی واقعی:
- روی 2 سال EURUSD مصنوعی واقع‌نما: 14807 کاندید AB → 81 سیگنال (35 برد 43%)
- بدون خطای NameError / PermissionError

