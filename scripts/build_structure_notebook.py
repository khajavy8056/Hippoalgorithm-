from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
old=nb.read(root/'HIPO_Pivot_Settlement.ipynb',as_version=4)
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
def embed(name):return f"(PACKAGE/'{name}.py').write_text({(root/'hippo_lab'/f'{name}.py').read_text()!r},encoding='utf-8')\n"
md('''# پیوت تسویه — نسخه مسیر قیمت، بدون الگوی شکل کندل

هسته روش: **AB قوی → BC اصلاحی → شکست B با CD کوتاه‌تر → برگشت قیمت و ورود معکوس**. تعداد کندل هم‌رنگ، فول‌بادی، نسبت سایه و FL حذف شده‌اند. موج‌ها با directional change و آستانه متناسب با نوسان گذشته تعریف می‌شوند.

**این نسخه هنوز سیستم پایدار یا سودآور اثبات‌شده نیست.** روی داده M1 سال۲۰۲۱ تنظیم میانه۵۹۵ تریگر داشت، ولی در نمونه تیک واقعی یک‌هفته‌ای با۱۱ معامله حدود۱٫۰۶٪ زیان داشت. ۲۰۲۱ فقط OHLC است؛ از آن بازده/استاپ‌ریت تولید نمی‌کنیم. دوره‌ها قبلاً دیده شده‌اند و holdout تازه نیستند.

نسخه shape-free هنوز **time-sampled** است: close یک‌دقیقه‌ای را می‌خواند، نه همه مسیر درون کندل؛ ادعای استقلال از تایم‌فریم نداریم. تبدیل مقیاس خطی قیمت تست شده، نه انتقال موفق بین DAX/Nasdaq/Dow/FX. ورودی مالی پیش‌فرض GBPUSD و حساب USD؛ اندازه قرارداد CFD/شاخص پشتیبانی نمی‌شود.
''')
cells.append(old.cells[1])
md('''## تعریف مشاهده‌پذیر و بدون بازنویسی گذشته

نوسان = EWMA قدرمطلق تغییر قیمت close، فقط تا مشاهده قبلی. آستانه = k×نوسان، در آغاز هر ساق موج **ثابت** می‌شود تا جهش نوسان معیار موج جاری را دستکاری نکند.

اکسترمم B/D در زمان وقوعش هنوز معلوم نیست؛ پس از برگشت قیمت به اندازه آستانه تأیید می‌شود. هر پیوت زمان وقوع و زمان تأیید جدا دارد؛ **تریگر فقط زمان تأیید D** است و اجرا روی تیک بعدی است، نه روی قله گذشته.

AB حداقل۲ آستانه و کارایی مسیر>=۰٫۴۵؛ BC اصلاح۲۰–۵۰٪؛ D فراتر ازB ولی CD<AB؛ تأیید D باید داخلB برگشته و هنوزC را نقض نکرده باشد. استاپ=D+بافر۰٫۱آستانه (قرینه برای خرید)، TP۲٫۵/۳R از ورود واقعی. اصلاح۶۵٪ فقط ablation جداست و تغییر نسبت به PDF محسوب می‌شود.

پنج تنظیم محدود، baseline از پیش k=۳؛ بهترین نتیجه هفته انتخاب نمی‌شود. افزایش سیگنال، خود اثبات کیفیت نیست. مدل به‌دلیل دیتای مالی کم آموزش داده نمی‌شود؛ حذف‌فقط‌باخت ممکن نیست تضمین شود.
''')
code('# @title ۲ — پیاده‌سازی مسیر قیمت و موتور واقعی اجرا\n'+''.join(embed(n) for n in ['engine','research','report','structural','pivot_long','pivot','pivot_experiment'])+'''
from hippo_lab.structural import StructureRules,detect_structure
from hippo_lab.engine import Execution,backtest
from hippo_lab.report import metrics
from hippo_lab.pivot_long import long_quote_gate
from dataclasses import replace,asdict
execution=Execution(pair=PAIR,account_currency=ACCOUNT_CURRENCY)
# Detection uses closed BID close; execution uses every real bid/ask quote.
prices=ticks.bid.resample('1min',closed='left',label='right').last().dropna()
prices=prices.loc[prices.index<=ticks.index[-1]]
VARIANTS=[('dc2',replace(StructureRules(),dc_multiple=2.)),('dc3_baseline',StructureRules()),
          ('dc4',replace(StructureRules(),dc_multiple=4.)),
          ('dc3_eff060',replace(StructureRules(),efficiency_min=.60)),
          ('dc3_retrace065',replace(StructureRules(),retrace_max=.65))]
print('Shape-free, but NOT sampling-frequency invariant')
''')
code('''# @title ۳ — آزمون علیت و مقیاس (مصنوعی فقط برای نرم‌افزار)
rng=np.random.default_rng(44)
fixture=pd.Series(100+np.cumsum(rng.normal(0,.1,5000)),index=pd.date_range('2020-01-01',periods=5000,freq='min',tz='UTC'))
e,_=detect_structure(fixture);ep,_=detect_structure(fixture.iloc[:3000])
pd.testing.assert_frame_equal(e.loc[e.index<=fixture.index[2999]],ep)
scaled,_=detect_structure(fixture*100+3000)
assert e.index.equals(scaled.index)
np.testing.assert_allclose(e.retrace,scaled.retrace,atol=1e-10)
print('PASS: prefix invariance, affine scale, causal confirmation')
''')
md('''## گزارش نهایی و حد شواهد

sample: آزمایش مهندسی روی کل هفته قبلاً دیده‌شده؛ نه OOS مستقل. local/histdata: حداقل یک سال quotes واقعی و پوشش روزکاری بررسی می‌شود؛ ۶۰٪ ابتدایی توسعه،۲۰٪ اعتبارسنجی و۲۰٪ نهایی بازپخش جدا. هیچ tuning یا انتخاب بهترین تنظیم با دیدن بخش نهایی انجام نمی‌شود؛ تغییر بعدی به داده آینده تازه نیاز دارد. این یک split اولیه است، نه چندfold و اثبات پایداری.

گزارش چهار سطح شواهد را جدا می‌کند: رویداد ساختاری، معاملات واقعی قابل اجرا، هزینه و DD، سودآوری/پایداری اثبات‌نشده. ML فعال نمی‌شود. parquet اکوییتی هر تیک ذخیره می‌شود؛ داده خام و خروجی حجیم در Git قرار نمی‌گیرند.
''')
code('''# @title ۴ — اجرای نسخه ساختاری و بک‌تست مالی با quote واقعی
out=Path('artifacts/structure');out.mkdir(parents=True,exist_ok=True)
mode='sample' if DATA_MODE=='sample' else 'long_research'
coverage=long_quote_gate(ticks,365) if mode!='sample' else {'warning':'Short reused sample; no long-term evidence'}
start,end=ticks.index[0],ticks.index[-1]
val_start=start+(end-start)*.6;test_start=start+(end-start)*.8
protocol={'variants':{n:asdict(r) for n,r in VARIANTS},'baseline':'dc3_baseline','execution':asdict(execution),
          'mode':mode,'data_audit':data_audit,'coverage':coverage,'threshold_selection':False,
          'splits':[str(val_start),str(test_start)],'ML_enabled':False,'stable_system':False}
protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,default=str).encode()).hexdigest()
(out/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2,default=str))
rows=[];funnel=[]
for name,rule in VARIANTS:
    events,log=detect_structure(prices,rule)
    events.to_csv(out/f'events_{name}.csv');log['pivots'].to_csv(out/f'pivots_{name}.csv',index=False)
    funnel.append({'variant':name,**log['counts']})
    periods=[('REUSED_SAMPLE_SMOKE',start,end+pd.Timedelta(nanoseconds=1))] if mode=='sample' else [
        ('development',start,val_start),('validation',val_start,test_start),('final_replay_NOT_for_selection',test_start,end+pd.Timedelta(nanoseconds=1))]
    for period,lo,hi in periods:
        quotes=ticks.loc[(ticks.index>=lo)&(ticks.index<hi)]
        if len(quotes)<2:raise ValueError('Missing quotes in period')
        sig=events.loc[(events.index>=lo)&(events.index<hi)]
        for rr in [2.5,3.]:
            s=sig.copy();s['rr']=rr
            result=backtest(quotes,s,execution)
            stats,months,daily,bootstrap=metrics(result)
            rows.append({'variant':name,'RR':rr,'period':period,'detected_triggers':len(sig),**stats})
            key=f'{name}_{rr:g}R_{period}'
            result['trades'].to_csv(out/f'trades_{key}.csv',index=False)
            months.to_csv(out/f'months_{key}.csv',index=False)
            result['equity'].to_frame().to_parquet(out/f'tick_equity_{key}.parquet')
            if name=='dc3_baseline' and rr==2.5:
                for stress,sc in [('spread1.5_slip2',replace(execution,spread_multiplier=1.5,slippage=execution.slippage*2)),('spread2_slip3',replace(execution,spread_multiplier=2.,slippage=execution.slippage*3))]:
                    rows.append({'variant':stress,'RR':rr,'period':period,'detected_triggers':len(sig),**metrics(backtest(quotes,s,sc))[0]})
table=pd.DataFrame(rows);table.to_csv(out/'comparison.csv',index=False)
pd.DataFrame(funnel).to_csv(out/'funnel.csv',index=False)
report={'status':'NOT_VERIFIED_STABLE_OR_PROFITABLE','protocol':protocol,
        'candle_shape_removed':True,'time_sampling_removed':False,
        'ML_enabled':False,'no_best_variant_selected':True}
(out/'summary.json').write_text(json.dumps(report,indent=2,default=str))
display(table)
display(pd.DataFrame(funnel))
html='<!doctype html><meta charset="utf-8"><h1>Price-path pivot — not proven stable</h1><p>More signals do not imply better accuracy. No best-of-test selection or ML success claim.</p>'+table.to_html(index=False)+pd.DataFrame(funnel).to_html(index=False)
(out/'report.html').write_text(html)
display(HTML(html))
import zipfile
with zipfile.ZipFile('HIPO_Structure_Report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in out.glob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.html']:z.write(p,arcname=p.name)
print('Report: HIPO_Structure_Report.zip; every-tick equity stays in artifacts/structure')
if 'google.colab' in sys.modules:
    from google.colab import files
    print('Download with files.download("HIPO_Structure_Report.zip")')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'colab':{'name':'HIPO_Price_Path.ipynb'}})
nb.validate(n);nb.write(n,root/'HIPO_Price_Path.ipynb');print('Built price-path notebook')
