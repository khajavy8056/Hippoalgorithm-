"""Standalone Colab deliverable for the PDF's pivot settlement logic only."""
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
def embed(name):
    return f"(PACKAGE / '{name}.py').write_text({(root/'hippo_lab'/f'{name}.py').read_text()!r}, encoding='utf-8')\n"
# Reuse only downloader/setup from existing generator, not its unrelated strategy hypotheses.
import runpy
runpy.run_path(str(root/'scripts/build_notebook.py'))
old=nb.read(root/'HIPO_Real_Tick_Research.ipynb',as_version=4)
first=old.cells[1].source.replace("bars = make_bars(ticks, '15min')","# M1 bid bars are created after loading the pivot module")
first=first.replace("print('M15 closed bars:',len(bars),'Quote RAM MB:',round(ticks.memory_usage(deep=True).sum()/1e6,1))", "print('Quote RAM MB:',round(ticks.memory_usage(deep=True).sum()/1e6,1))")
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# پیوت تسویه — از PDF فارسی تا پژوهش تیک‌به‌تیک

فقط منطق PDF «ستاپ پیوت تسویه جلیل ضرغام» بررسی می‌شود، نه راهبردهای نامرتبط. **AB صعودی → فروش بعد از تریگر معکوس؛ AB نزولی → خرید. شکست B به‌تنهایی ورود نیست.**

انتخاب شما: هر دو برداشت سایه/کلوز، TP ثابت ۲٫۵/۳R، فعلاً تست با داده موجود. ۲۰ تصویر یکتا DAX/FXCM یک‌دقیقه‌ای ساختاری بررسی شدند. فایل‌های با نام 04-26 و 05-05 در تصویر تاریخ 04-23 و 05-06 دارند. یک تصویر تکراری است.

**تطبیق عددی سیگنال تصاویر DAX تأیید نشده**: داده خام همان نماد و ساعت نمودار موجود نیست. نمونه اجرا GBPUSD دارای Bid/Ask است، نه DAX. تصاویر منتخب و R نام فایل‌ها برچسب یادگیری یا مدرک نرخ برد نیستند. هدف ماهانه ۳–۴٪ و DD<۱۰٪ هنوز اثبات نشده است.

نوت‌بوک خودکفا؛ همه سلول‌ها را به ترتیب اجرا کنید. پیش‌فرض sample فقط آزمون مهندسی است. برای تاریخچه خودتان local یا histdata را انتخاب کنید. اکوییتی و هزینه‌ها هر تیک اجرا می‌شوند؛ داده خام چندساله RAM زیادی می‌خواهد.
''')
code(first)
md((root/'reports/pivot/logic_spec_fa.md').read_text())
code('# @title ۲ — موتور اجرا، قواعد پیوت و ابزار تطبیق\n'+''.join(embed(n) for n in ['engine','research','report','pivot','pivot_experiment','pivot_match'])+'''
from hippo_lab.engine import Execution,backtest
from hippo_lab.pivot import PivotRules,pivot_bars,detect_pivots,FEATURE_COLUMNS
from hippo_lab.pivot_experiment import run_pivot_experiment
from hippo_lab.pivot_match import read_chart_bars,compare_images
execution=Execution(pair=PAIR,account_currency=ACCOUNT_CURRENCY)
bars=pivot_bars(ticks,'1min')
print('Closed BID M1 bars:',len(bars))
print(execution)
print(PivotRules())
# DAX/CFD contract sizing is NOT implemented in this FX account engine.
# Costs/account units must be set before outcomes. No leverage tuning to reach target.
''')
md('''## آزمون علیت و حسابداری

مصنوعی زیر فقط fixture نرم‌افزار است و هیچ بازده آن نتیجه مالی نیست. روی همین داده باید AB/BC/شکست B/تریگر به ترتیب تشخیص داده شود، سیگنال فروش فقط بعد از شکست کف سیگنال صادر شود و افزودن آینده نتیجه گذشته را عوض نکند.
''')
code('''# @title ۳ — آزمون بازتولیدپذیری منطق
from dataclasses import replace
rows=[(100,100.5,99.5,100)]*8
rows += [(100,102.2,99.8,102),(102,104.2,101.8,104),(104,106.2,103.8,106),
         (106,106.1,105,105.5),(105.5,105.8,104.8,105),(105,105.2,104,104.5),
         (104.5,107,104.3,106.8),(106.8,107,103.5,103.8),(103.8,104,103,103.2)]
ix=pd.date_range('2024-01-02 14:00',periods=len(rows),freq='1min',tz='UTC')
fixture=pd.DataFrame(rows,columns=['open','high','low','close'],index=ix)
check_rule=replace(PivotRules(),atr_period=2,ab_max_bars=3,ab_min_median_range_atr=.1,
                   signal_range_atr=.1,trigger_body_atr=.1)
events,trace=detect_pivots(fixture,check_rule)
assert len(events)==1 and events.iloc[0].side==-1 and events.index[0]==ix[15]
assert events.iloc[0].signal_time==ix[14] and events.iloc[0].bc_bars==3
past,_=detect_pivots(fixture.iloc[:16],check_rule)
pd.testing.assert_frame_equal(events,past)
print('PASS: PDF phase ordering, strict reversal timing, BC duration, prefix invariance')
''')
# Embed image audit, not raw PNG/PDF in notebook; files remain accessible in Doc repo.
import pandas as pd
annotations=pd.read_csv(root/'reports/pivot/image_review.csv').to_json(orient='records')
md('''## تطبیق تصاویر: مرز شواهد

جدول زیر برداشت دستی همه نمونه‌هاست. `QUALITATIVE_ONLY_NOT_NUMERIC_MATCH` یعنی **کد روی داده همین تصاویر اجرا نشده**. برای مقایسه واقعی، CSV خروجی OHLC همان FXCM_GER30 با زمان و منطقه زمانی لازم است. ابزار تطبیق اختیاری در همین نوت‌بوک وجود دارد؛ نزدیکی زمان/جهت تنها تست اولیه است و اثبات یکسان‌بودن A/B/C/D نیست.
''')
code('# @title ۴ — مرور تصاویر و تطبیق اختیاری داده همان نمودار\n'+f"image_annotations=pd.DataFrame(json.loads({annotations!r}))\n"+'''
display(image_annotations[['image','chart_date','approx_trigger_chart_local','side','structure_observed','caveat','audit_status']])
CHART_CSV = '' # @param {type:'string'}
CHART_TIMEZONE = 'Europe/Berlin' # @param {type:'string'}
CHART_TIMESTAMP_IS_OPEN = True # @param {type:'boolean'}
# Default timezone above is ONLY a placeholder input, NOT known screenshot timezone.
# Fill CHART_CSV only with actual matching symbol data and correct chart timezone.
if CHART_CSV:
    chart_bars=read_chart_bars(CHART_CSV,CHART_TIMEZONE,CHART_TIMESTAMP_IS_OPEN)
    comparisons=[]
    for boundary in ['wick','close']:
        match,chart_events,chart_trace=compare_images(chart_bars,image_annotations,CHART_TIMEZONE,
                                                     replace(PivotRules(),boundary=boundary))
        comparisons.append(match)
    chart_comparison=pd.concat(comparisons,ignore_index=True)
    display(chart_comparison)
    Path('artifacts/pivot').mkdir(parents=True,exist_ok=True)
    chart_comparison.to_csv('artifacts/pivot/image_OHLC_match.csv',index=False)
else:
    print('Exact DAX screenshot reproduction: NOT VERIFIED — no matching OHLC supplied')
''')
md('''## پروتکل آزمایش و مدل کاهش خطا

چهار نسخه **همین منطق**: سایه/کلوز × TP۲٫۵/۳R. baseline از پیش `wick_2.5R` است؛ بهترین نتیجه تست انتخاب نمی‌شود. واگرایی زمانی فقط روی development به‌صورت تشخیصی گزارش می‌شود. تقسیم زمانی ۶۰٪ آموزش، ۲۰٪ اعتبارسنجی، ۲۰٪ آزمون؛ این اجرای اولیه exploratory است، نه ارزیابی چند fold. برای تأیید نهایی چندساله، بازار دیگر و >=۲۴ ماه OOS لازم است.

ML برچسب معاملات مستقل تیک‌به‌تیک همین ستاپ را می‌خواند؛ فقط ویژگی‌های قابل مشاهده در تریگر، نه R تصویر یا آینده. آموزش با exposure غیرهم‌پوشان و embargo۲۴ساعت؛ حداقل ۱۲ماه ورودی، ۱۰۰ label آموزش، ۳۰ برد و ۳۰ باخت اعتبارسنجی. انتخاب آستانه با حفظ >=۹۵٪ تعداد و پول بردها، حذف >=۱۰٪ باخت‌ها و PnL بهتر. **این قید تضمین حفظ برد آینده نیست.**

آستانه و ضرایب قبل از بک‌تست نهایی در `frozen_before_test.json` قفل می‌شوند. در نمونه کوتاه ML bypass است؛ غیرفعال بودن به‌معنای بهترشدن سیستم نیست. گزارش paired می‌گوید چند برد/باخت و چه مقدار سود حذف شده؛ گزارش حساب مشترک جداست چون حذف یک معامله ممکن است معامله بعدی را ممکن کند.

سلول آخر: گزارش تمامی نسخه‌ها، تنش هزینه، funnel ابطال‌ها، ردگیری A/B/C/D و علت‌ها، اکوییتی هر تیک، معاملات و R خالص، ماه‌های کامل/ناقص، نمودار ستاپ‌های **واقعی GBPUSD** و گزارش HTML. مسیر `artifacts/pivot`. فایل equity بزرگ در ZIP گزارش قرار نمی‌گیرد.
''')
code('''# @title ۵ — اجرای پژوهش، بک‌تست واقعی و گزارش جامع
mode='sample' if DATA_MODE=='sample' else 'research'
report,comparison,results=run_pivot_experiment(ticks,data_audit,execution,
                                             folder='artifacts/pivot',mode=mode,bars=bars)
Path('artifacts/pivot/image_review.csv').write_text(image_annotations.to_csv(index=False))
print(json.dumps(report,indent=2,ensure_ascii=False,default=str))
display(comparison)
display(HTML(Path('artifacts/pivot/report.html').read_text()))
import zipfile
with zipfile.ZipFile('HIPO_Pivot_Report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in Path('artifacts/pivot').glob('*'):
        if p.is_file() and p.suffix in ['.csv','.json','.html','.png']:z.write(p,arcname=p.name)
print('Report ZIP: HIPO_Pivot_Report.zip')
print('Every-tick equity: artifacts/pivot/test_tick_equity_*.parquet')
if 'google.colab' in sys.modules:
    from google.colab import files
    print('Download with: files.download("HIPO_Pivot_Report.zip")')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'HIPO_Pivot_Settlement.ipynb'}})
nb.validate(n);nb.write(n,root/'HIPO_Pivot_Settlement.ipynb');print('Built pivot notebook',len(cells),'cells')
