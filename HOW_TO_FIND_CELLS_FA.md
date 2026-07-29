# کجا سلول‌های کامل هستند؟

## مسیرهای اصلی (هر کدام کامل و آماده paste در Colab)

### سلول فیچرها
1. `CELLS/01_FEATURE_ENGINE.py`
2. `CELL_01_FEATURE_ENGINE.py`  (ریشه پروژه)
3. `DELIVERABLE_CELL_01_FEATURE_ENGINE_COMPLETE.py`

### سلول لیبل‌گذاری + GA + گزارش کیفیت
1. `CELLS/02_LABELING_FORGE.py`
2. `CELL_02_LABELING_FORGE.py`  (ریشه پروژه)
3. `DELIVERABLE_CELL_02_LABELING_FORGE_COMPLETE.py`

## ترتیب اجرا
1. Downloader
2. سلول فیچر
3. سلول لیبل (دستی یا GA → Apply → تب کیفیت فیچر)
4. Training

## GA هدف
- Wins > Losses (قید سخت)
- Feature Quality بالا
- تعداد نمونه کافی
