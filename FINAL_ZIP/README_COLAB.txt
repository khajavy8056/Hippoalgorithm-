HIPO FINAL CELLS — Feature Quality + Multi-Obj Labeling (2026-07-29)
====================================================================

ترتیب اجرا در Google Colab:

  0) Downloader (سلول اصلی پروژه) → Tick parquet در /content/hipo_lab_data
  1) 01_FEATURE_ENGINE_FINAL.py
       → v25 PIVOT-ALIGNED (PSA_/LSW_/FVG_ + shift1 + killzone)
  2) 02_LABELING_FORGE_FINAL_COLAB_CLEAN.py
       → Manual یا تب GA چندهدفه v4 (Sample + WinRate + FeatureQuality)
       → پیشنهاد شروع: min_samples=100, RR=1.0, EUR spread=0.00012 / XAU=0.35
       → بهترین کانفیگ تست‌شده: swing_n=1, ab_max=12, atr_min=1.1, noise=0.5, bc_max=0.65, RR=1.0
  3) 00_FEATURE_LABEL_QUALITY_LAB.py   ← جدید، اجباری قبل از Train
       → می‌سازد: selected_features.json + گزارش HTML
       → اگر Global Score < 45 → Train نکن
  4) 03_TRAINING_FINAL.py
       → v27: خودکار selected_features را می‌خواند
       → Quality Gate: Score<45 را قطع می‌کند

نتیجه اندازه‌گیری‌شده روی EURUSD 2y تست:
  labels 81→137 | Quality 71→88 | shuffle p 0.45→0.09 | OOS PR-AUC 0.38→0.63
  Precision@0.5 ≈71% ولی فقط 7 ترید OOS (CI پهن) → برای 60% پایدار دیتای بیشتر لازم

جزئیات: ../FEATURE_QUALITY_LAB/README_FA.md و FINAL_RESULT.md
