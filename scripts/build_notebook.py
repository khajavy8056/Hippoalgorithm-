"""Build a standalone Colab notebook embedding the exact tested library sources."""
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
def embed(name):
    text=(root/'hippo_lab'/f'{name}.py').read_text()
    return f"(PACKAGE / '{name}.py').write_text({text!r}, encoding='utf-8')\n"
md('''# HIPO — آزمایش قابل ممیزی روی تیک واقعی

**هدف پژوهش:** متوسط سود خالص ماهانه ۳–۴٪ و افت سرمایه اکوییتی زیر ۱۰٪ روی یک جفت ارز. این نسخه **استراتژی موفق اثبات‌شده نیست**؛ داده در دسترس این اجرا یک هفته است و هدف را تأیید نمی‌کند. افزایش اهرم برای نمایش سود هدف ممنوع است.

نوت‌بوک خودکفا است؛ فایل Python جدا یا clone لازم ندارد. حالت پیش‌فرض `sample` برای بررسی اجراست. برای پژوهش چندساله از `histdata` یا `local` استفاده کنید، پارامترها را **پیش از دیدن نتیجه** تثبیت کنید و همه سلول‌ها را به ترتیب اجرا کنید.

[مخزن GitHub](https://github.com/khajavy8056/Hippoalgorithm-) · داده نمونه: آینه عمومی GAIN Capital، GBPUSD، هفته اول آوریل ۲۰۱۶؛ صحت منبع مستقلاً توسط بروکر تأیید نشده است.
''')
first='''# @title ۱ — دریافت تیک واقعی (همان روش token + POST فایل ارسالی)
import sys, subprocess, importlib.util, json, glob, hashlib
from pathlib import Path
requirements = ['numpy>=2.2,<2.5', 'pandas>=2.2,<3.1', 'pyarrow>=18,<26',
                'numba>=0.61,<0.69', 'scikit-learn>=1.6,<1.10', 'matplotlib>=3.9,<3.12',
                'requests>=2.32,<3', 'beautifulsoup4>=4.12,<5']
modules = ['numpy','pandas','pyarrow','numba','sklearn','matplotlib','requests','bs4']
if any(importlib.util.find_spec(m) is None for m in modules):
    subprocess.check_call([sys.executable,'-m','pip','install',*requirements])
import numpy as np, pandas as pd
from IPython.display import display, HTML
PACKAGE = Path('.hippo_notebook') / 'hippo_lab'
PACKAGE.mkdir(parents=True, exist_ok=True)
(PACKAGE / '__init__.py').write_text('')
sys.path.insert(0, str(PACKAGE.parent.resolve()))
# Re-running all cells reloads the embedded implementation, not stale imports.
for name in list(sys.modules):
    if name == 'hippo_lab' or name.startswith('hippo_lab.'):
        del sys.modules[name]
'''+embed('data')+'''
from hippo_lab.data import download_histdata, download_sample, read_ticks, make_bars, validate_ticks, HISTDATA_TZ
DATA_MODE = 'sample' # @param ['sample', 'histdata', 'local']
PAIR = 'GBPUSD' # @param {type:'string'}
ACCOUNT_CURRENCY = 'USD' # @param {type:'string'}
START_YEAR = 2022 # @param {type:'integer'}
END_YEAR = 2024 # @param {type:'integer'}
LOCAL_GLOB = '/content/drive/MyDrive/HIPO_DATA/GBPUSD_TICK_*.parquet' # @param {type:'string'}
MOUNT_DRIVE = False # @param {type:'boolean'}
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

if DATA_MODE == 'sample':
    if PAIR != 'GBPUSD': raise ValueError('The pinned sample is GBPUSD only')
    ticks, data_audit = download_sample()
else:
    files = []
    if DATA_MODE == 'histdata':
        now = pd.Timestamp.now(tz='UTC')
        for year in range(START_YEAR, END_YEAR+1):
            for month in range(1,13):
                if pd.Timestamp(year=year,month=month,day=1,tz='UTC')+pd.DateOffset(months=1)>now:
                    continue # skip incomplete current month
                print(f'Downloading {PAIR} {year}-{month:02d}', flush=True)
                files.append(download_histdata(PAIR,year,month))
    elif DATA_MODE == 'local':
        files = sorted(Path(p) for p in glob.glob(LOCAL_GLOB))
    if not files: raise ValueError('No quote files. Mount Drive or correct LOCAL_GLOB.')
    frames=[]; audits=[]; manifest=[]
    for path in files:
        f,a=read_ticks(path, naive_tz=HISTDATA_TZ)
        # Annual files from original downloader use Bid/Ask and fixed EST naive index.
        frames.append(f); audits.append(a)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        manifest.append({'file':str(path),'sha256':h.hexdigest()})
    ticks, data_audit=validate_ticks(pd.concat(frames))
    del frames
    # Overlapping files are dangerous: equal timestamps with different quotes may be legitimate
    # inside one file, but cross-file overlaps must be corrected by the user.
    ranges=sorted((pd.Timestamp(a['start']),pd.Timestamp(a['end'])) for a in audits)
    if any(b[0]<=a[1] for a,b in zip(ranges,ranges[1:])):
        raise ValueError('Overlapping quote files; choose non-overlapping monthly OR annual files, not both')
    data_audit.update(source='HistData quote ZIP / user original parquet',pair=PAIR,
                      file_audits=audits, manifest=manifest, timezone='HistData fixed EST UTC-5')
if len(ticks)>100_000_000:
    print('WARNING: >100M quotes. High-RAM Colab required; keep only one pair. No downsampling of execution quotes.')
bars = make_bars(ticks, '15min')
print(json.dumps(data_audit, indent=2, ensure_ascii=False))
print('M15 closed bars:',len(bars),'Quote RAM MB:',round(ticks.memory_usage(deep=True).sum()/1e6,1))
display(ticks.head())
'''
code(first)
md('''## ۲ — قواعد اجرای واقعی‌تر و محدودیت‌ها

- خرید در **Ask** و فروش در **Bid**؛ سیگنال فقط با کندل بسته‌شده، ورود روی تیک بعدی پس از تأخیر.
- کمیسیون هر سمت، لغزش نامطلوب، اسپرد مشاهده‌شده، حد اهرم و ریسک هر معامله ۰٫۲۵٪.
- استاپ در گپ با قیمت مشاهده‌شده پر می‌شود؛ سود حدی با عبور قیمت و بدون بهبود قیمت پر می‌شود. دفتر سفارش و عمق بازار در این داده نیست.
- اکوییتی شامل سود/زیان باز و هزینه خروج در **هر تیک**؛ توقف افت سرمایه ۸٪ و زیان روزانه ۲٪، **بدون تضمین سقف** در گپ.
- فقط یک پوزیشن؛ خروج پیش از rollover نیویورک، بدون فرض سوآپ صفر برای پوزیشن شبانه. اگر داده قطع شود خروج در اولین تیک بعدی است و علامت گپ ثبت می‌شود.
- حساب باید ارز مظنه باشد؛ مثال GBPUSD با USD. این نسخه تبدیل ارز برای حساب USD روی USDJPY یا طلا را پیاده نمی‌کند.
- ساعت HistData، EST ثابت UTC−5 است (بدون DST). فرض ساعت نمونه GAIN، نیویورک است و در گزارش ثبت می‌شود.
''')
code('# @title ۲ — موتور تیک‌به‌تیک و تنظیم ریسک\n'+embed('engine')+'''
from hippo_lab.engine import Execution, backtest
execution = Execution(pair=PAIR, account_currency=ACCOUNT_CURRENCY)
# تغییر هزینه‌ها فقط قبل از مشاهده آزمون نهایی؛ برای EURUSD/JPY واحد قیمت لغزش را درست تنظیم کنید.
print(execution)
''')
md('''## ۳ — فرضیه‌ها و پروتکل ضد بیش‌برازش

دو شکست کانال، دو بازگشت به میانگین در رژیم کم‌روند، و دو آستانه برای Logistic Regression با مقیاس‌بندی فقط روی آموزش. مدل عمداً ساده است: پیچیدگی بیشتر الزاماً مزیت معاملاتی نیست. برچسب ML جهت بازده ۸ کندل بعد است، نه سود قابل معامله؛ **معیار انتخاب بازده خالص بک‌تست است نه AUC**.

آموزش ML با برچسب‌های غیرهم‌پوشان، پایان برچسب پیش از مرز و embargo زمانی ۲۴ ساعت. داده تست برای early stopping یا تغییر آستانه مصرف نمی‌شود. ویژگی‌ها فقط گذشته و کندل فعلی بسته‌شده؛ هیچ pivot با اطلاع آینده نداریم.

**حالت کامل:** حداقل ۲۴ ماه کامل ورودی؛ پنجره ۱۲ ماه آموزش + ۳ ماه اعتبارسنجی + ۳ ماه آزمون جلو‌رونده، ۶ ماه آخر مستقل و دست‌نخورده. برای حداقل ۴ fold جلو‌رونده حدود ۳۳ ماه ورودی لازم است؛ برای ۲۴ ماه OOS ورودی بیشتری لازم است. نمونه یک‌هفته‌ای فقط smoke test است و ML عمداً با شواهد ناکافی غیرفعال می‌شود.

گزینه با کمتر از ۳۰ معامله اعتبارسنجی، بازده خالص غیرمثبت، یا DD بالای ۸٪ رد می‌شود؛ در نبود گزینه، **CASH / عدم معامله**. هیچ بازتنظیمی پس از نتیجه holdout انجام نمی‌شود. انتخاب و ضرایب مدل قبل از اجرای holdout در `artifacts/frozen_selection.json` با هش ذخیره می‌شوند.
''')
code('# @title ۳ — ویژگی‌ها، مدل و انتخاب زمانی\n'+embed('research')+'''
from hippo_lab.research import research_run, CANDIDATES, PROTOCOL, protocol_hash, features, signals_for, fit_candidate
print('Protocol hash:', protocol_hash(execution))
print(json.dumps(PROTOCOL,indent=2,ensure_ascii=False))
display(pd.DataFrame(CANDIDATES))
''')
md('''## ۴ — بررسی علیت قبل از آزمایش

این تست کوچک **مصنوعی و فقط تست نرم‌افزار** است؛ به عنوان نتیجه مالی گزارش نمی‌شود. با دستکاری آینده، ویژگی‌ها و مدل آموزش‌دیده در گذشته نباید تغییر کنند. تست‌های بیشتر موتور در پوشه `tests` مخزن هستند.
''')
code('''# @title ۴ — تست نشت اطلاعات و حسابداری
rng=np.random.default_rng(8056)
ix=pd.date_range('2020-01-01',periods=1800,freq='15min',tz='UTC')
close=1.2+np.cumsum(rng.normal(0,.0004,len(ix)))
fixture=pd.DataFrame({'open':close,'high':close+.0002,'low':close-.0002,'close':close,'spread':.0001},index=ix)
mutated=fixture.copy(); mutated.loc[ix[1300]:,'close']*=5
np.testing.assert_allclose(features(fixture).iloc[:1300],features(mutated).iloc[:1300],equal_nan=True)
m1=fit_candidate(CANDIDATES[4],fixture,ix[0],ix[1300]); m2=fit_candidate(CANDIDATES[4],mutated,ix[0],ix[1300])
assert not isinstance(m1,str)
np.testing.assert_allclose(m1[-1].coef_,m2[-1].coef_)
np.testing.assert_allclose(m1[0].mean_,m2[0].mean_)
check_index=pd.date_range('2024-01-02 14:00',periods=4,freq='s',tz='UTC')
check_ticks=pd.DataFrame({'bid':[1.2]*4,'ask':[1.2001]*4},index=check_index)
check_signals=pd.DataFrame({'side':[1],'distance':[.001],'rr':[1.]},index=check_index[:1])
check=backtest(check_ticks,check_signals,execution)
assert abs(check['trades'].pnl.sum()-(check['equity'].iloc[-1]-execution.initial_cash))<1e-7
assert check['trades'].entry_time.iloc[0]>check_index[0]
print('PASS: causal features, purged ML, executable spread + commission reconciliation')
del fixture,mutated,check_ticks,check_signals
''')
md('''## ۵ — اجرا و گزارش نهایی

آخرین سلول اجرای واقعی پژوهش و گزارش جامع را انجام می‌دهد. `sample` را با سود ماهانه مقایسه نکنید. تغییر پارامترها پس از مشاهده holdout آن را به داده اعتبارسنجی تبدیل می‌کند؛ برای آزمون بعدی تاریخ آینده جدید لازم است.

گزارش: تمام آزمایش‌های اعتبارسنجی، معاملات و R خالص، بازده ماه‌های کامل و ناقص، اکوییتی تیک‌به‌تیک، DD، گپ‌های داده، حسابداری هزینه، سناریوهای تنش ثابت، نمودار و بازنمونه‌گیری بلوکی فقط با تاریخچه کافی. تنش‌ها همان سیگنال‌های قفل‌شده را بازپخش می‌کنند؛ انتخاب مجدد انجام نمی‌شود. ارزیابی بازار دیگر باید **بدون انتخاب پس از دیدن نتیجه** و با ارز حساب/هزینه صحیح جداگانه اجرا شود.

افت سرمایه تاریخی و bootstrap تضمین افت آینده نیست. این بک‌تستر شبیه‌سازی quote-level است، نه اثبات قابلیت اجرای سفارش در حجم دلخواه. تاریخچه چندساله در حافظه نگهداری می‌شود؛ برای یک جفت ارز پرحجم از Colab High-RAM استفاده کنید. داده خام در Git ذخیره نمی‌شود.
''')
code('# @title ۵ — بک‌تست تیک‌به‌تیک، آزمون نهایی و گزارش\n'+embed('report')+'''
from hippo_lab.report import write_report, metrics
mode = 'sample' if DATA_MODE == 'sample' else 'full'
run = research_run(ticks,bars,execution,mode=mode)
diagnostics = None
if mode == 'sample':
    rows=[]
    for candidate in CANDIDATES:
        model=fit_candidate(candidate,bars,ticks.index[0],pd.Timestamp('2016-04-07',tz='America/New_York').tz_convert('UTC'))
        signals=signals_for(candidate,bars,ticks.index[0],ticks.index[-1],model)
        diagnostic=backtest(ticks,signals,execution)
        rows.append({'candidate':candidate['name'],'role':'SMOKE ONLY — NOT OOS',**metrics(diagnostic)[0]})
    diagnostics=pd.DataFrame(rows)
summary=write_report(run,data_audit,folder='artifacts',diagnostics=diagnostics)
print(json.dumps(summary,indent=2,ensure_ascii=False))
display(run['selection'])
if diagnostics is not None: display(diagnostics)
display(HTML(Path('artifacts/report.html').read_text()))
print('Files: artifacts/report.html, summary.json, holdout_trades.csv, holdout_months.csv, equity_every_tick.parquet')
# Optional: download a compact report ZIP; tick equity can be very large, so exclude it.
import zipfile
with zipfile.ZipFile('HIPO_report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in Path('artifacts').glob('*'):
        if p.is_file() and p.suffix in ['.json','.html','.png','.csv']: z.write(p,arcname=p.name)
if importlib.util.find_spec('google.colab'):
    from google.colab import files
    display(HTML('<p>گزارش در HIPO_report.zip آماده است؛ برای دریافت: files.download("HIPO_report.zip")</p>'))
''')
# find_spec on google.colab raises if no google package; use try in standalone environments.
cells[-1].source=cells[-1].source.replace("if importlib.util.find_spec('google.colab'):","if 'google.colab' in sys.modules:")
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'HIPO_Real_Tick_Research.ipynb'}})
nb.validate(notebook)
nb.write(notebook,root/'HIPO_Real_Tick_Research.ipynb')
print('Built standalone notebook:',len(cells),'cells')
