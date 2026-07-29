# HIPO — Feature Quality + Multi-Objective Labeling + Selected Training

**تاریخ:** 2026-07-29  
**وضعیت:** روی دیتای تست EURUSD (2 سال) end-to-end اجرا و اندازه‌گیری شد.

---

## نتیجه واقعی pipeline (نه حدس)

| معیار | قبل (baseline) | بعد (این کار) |
|--------|----------------|----------------|
| تعداد سیگنال | 81 | **137** (۱.۶۹×) |
| WinRate | 43.2% | 40.9% |
| Global Feature Quality | 71.2 | **87.6** |
| Probe CV PR-AUC | 0.52 | **0.64** |
| Label-Shuffle p | 0.45 (نویز) | **0.09 ✅ واقعی** |
| OOS PR-AUC (selected) | 0.38 | **0.63** |
| Lift vs base OOS | 0.83 | **1.32** |
| Precision@0.5 OOS | ~46% | **71%** (n=7، CI هنوز پهن) |

### بهترین پیکربندی لیبل
`swing1_loose_rr1`:
- swing_n=1, ab_max=12, ab_atr_min=1.1, noise=0.5, bc_retrace_max=0.65  
- RR=**1.0**, max_bars=60, spread EUR=0.00012  
- Fitness چندهدفه: **64.35** (نمونه + WR + FeatureQuality)

> نکته: کانفیگ `ab_gt_cd_rr1` با 882 سیگنال و WR 45% آمد ولی FeatureQuality فقط 14 → فیتنس چندهدفه درست آن را **برنده نکرد**. این دقیقاً هدف سیستم جدید است.

---

## سلول‌های Colab (ترتیب اجرا)

| # | فایل | کار |
|---|------|-----|
| 1 | `FINAL_ZIP/01_FEATURE_ENGINE_FINAL.py` | Feature Engine **v25** + PSA/LSW/FVG |
| 2 | `FINAL_ZIP/02_LABELING_FORGE_FINAL_COLAB_CLEAN.py` | Labeling + **GA چندهدفه v4** |
| 3 | `FINAL_ZIP/00_FEATURE_LABEL_QUALITY_LAB.py` | **Quality Lab** (جدید) |
| 4 | `FINAL_ZIP/03_TRAINING_FINAL.py` | Sniper **v27** — می‌خواند `selected_features.json` + Quality Gate |

### قانون طلایی
```
اگر Quality Lab → Global Score < 45  →  Training متوقف (v27 خودش هم قطع می‌کند)
اگر Score ≥ 65 و shuffle p < 0.15   →  برو Training
```

---

## چه چیزی ساخته شد؟

### 1) فیچرهای هم‌راستا با لیبل (v25)
- `PSA_*` : AB/BC/Sweep/Box/CD readiness  
- `LSW_*` : liquidity sweep + tick align  
- `FVG_*` : Fair Value Gap  
همه `shift(1)` — leak-check پاس شد.

### 2) Quality Lab
برای هر فیچر: AUC / MI / |r| → KEEP_STRONG/KEEP/WEAK/DROP  
برای کل سیستم: Global Score + Label Shuffle + لیست selected.

### 3) Optimizer چندهدفه
```
Fitness = w_count·Samples + w_winrate·Class1% + w_feat·FeatureQuality
```
دیگر «نمونه زیاد ولی غیرقابل‌یادگیری» یا «WR بالا با n=50» برنده نمی‌شود.

### 4) Training v27
- اگر `selected_features.json` باشد → فقط همان فیچرها  
- اگر Score < 45 → Training را قطع می‌کند  

---

## محدودیت باقی‌مانده (صادقانه)

- Precision 71% روی **فقط 7 معامله OOS** است → Wilson CI پایین ≈36% → **هنوز ادعای 60% پایدار نیست**.
- برای ادعای جدی 60% نیاز است: **≥100 معامله OOS** (چند نماد / چند سال بیشتر).
- با RR=2.5 نمونه و edge سخت‌تر می‌شود؛ برای نمایش precision از RR=1.0–1.5 شروع کنید.

---

## خروجی‌های اندازه‌گیری‌شده در ریپو

```
pipeline_out/FINAL_RESULT.md
pipeline_out/FINAL_RESULT.json
pipeline_out/labeling_leaderboard.csv
pipeline_out/feature_quality_table.csv
pipeline_out/selected_features.json
pipeline_out/hipo_lab_data/EURUSD_Labeled.parquet   # 137 signals
```

اجرای مجدد محلی:
```bash
.venv/bin/python FEATURE_QUALITY_LAB/pipeline_end_to_end.py
```
