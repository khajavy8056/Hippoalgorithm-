# تحویل دو سلول کامل — Feature + Labeling

## فایل‌های اصلی (کپی مستقیم در Colab)

| سلول | فایل |
|------|------|
| **۱. فیچرها** | `DELIVERABLE_CELL_01_FEATURE_ENGINE_COMPLETE.py` |
| **۲. لیبل‌گذاری + GA + کیفیت** | `DELIVERABLE_CELL_02_LABELING_FORGE_COMPLETE.py` |

همین محتوا در `FINAL_ZIP/` و `FINAL_DELIVERABLE/` هم هست.

---

## سلول ۱ — Feature Engine v25 COMPLETE

**ورودی:** `*_Tick*.parquet`  
**خروجی:** `*_Features.parquet`

شامل:
- تمام گروه‌های ساختار / روند / نوسان / نقدینگی / زمان (Killzone)
- **PSA_*** هم‌راستا با Pivot Settlement
- **LSW_*** Liquidity Sweep
- **FVG_*** Fair Value Gap
- همه با `shift(1)` + warmup 250 + dropna (بدون fillna(0))

---

## سلول ۲ — Labeling Forge v5 COMPLETE

### تب ۱: استخراج دستی
پارامترهای کامل Pivot Settlement + tick race

### تب ۲: اپتیمایز ژنتیک v5
اهداف فیتنس:
1. **Wins > Losses** (قید سخت — اگر برقرار نباشد فیتنس ≈ -500000)
2. **Feature_Quality** بالا
3. **تعداد نمونه** کافی

### تب ۳: گزارش کیفیت فیچر ↔ لیبل
- رتبه هر فیچر (AUC / MI / Quality → KEEP/DROP)
- Global Score + Label-Shuffle
- ذخیره `selected_features.json` برای Training
- گزارش HTML

---

## ترتیب اجرا

```
Downloader → سلول۱ Features → سلول۲ Labeling
  ├─ Manual یا GA
  ├─ Apply best (اگر GA)
  └─ تب کیفیت فیچر
→ Training (با selected_features.json)
```

### تنظیمات پیشنهادی شروع GA
- min_samples: 80–150 (۲ سال)
- RR: 1.0
- EUR spread: 0.00012 | XAU: 0.35
- population 20 × generations 12

### کانفیگ تست‌شده خوب (EURUSD)
swing_n=1, ab_max=12, ab_atr=1.1, noise=0.5, bc_max=0.65, RR=1.0, max_bars=60
