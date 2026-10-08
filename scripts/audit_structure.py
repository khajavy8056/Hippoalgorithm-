"""Predeclared price-path sensitivity study. Historical periods already inspected: NOT virgin holdout."""
from pathlib import Path
from dataclasses import replace,asdict
import json,hashlib
import pandas as pd
from scripts.audit_pivot_year import load_year
from hippo_lab.structural import StructureRules,detect_structure
from hippo_lab.data import download_sample
from hippo_lab.pivot import pivot_bars
from hippo_lab.engine import backtest,Execution
from hippo_lab.report import metrics

VARIANTS=[('dc2',replace(StructureRules(),dc_multiple=2.)),('dc3_baseline',StructureRules()),
          ('dc4',replace(StructureRules(),dc_multiple=4.)),
          ('dc3_eff060',replace(StructureRules(),efficiency_min=.60)),
          ('dc3_retrace065',replace(StructureRules(),retrace_max=.65))]

def main():
    out=Path('reports/structure');out.mkdir(parents=True,exist_ok=True)
    bars,audit=load_year();ticks,tick_audit=download_sample();sample_bars=pivot_bars(ticks);cfg=Execution()
    protocol={'variants':{n:asdict(r) for n,r in VARIANTS},'data':audit,'execution':asdict(cfg),
              'no_best_variant_selection':True,'2021_role':'REUSED_STRUCTURAL_RESEARCH_NOT_FINANCIAL',
              '2016_role':'REUSED_WEEK_ENGINEERING_NOT_UNTOUCHED_OOS',
              'sample_rr_variants':[2.5,3.],'rule_trials':5,'sample_replay_trials':10}
    protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];months=[];smokes=[];figures=[]
    for name,rule in VARIANTS:
        print(name,flush=True)
        e,log=detect_structure(bars.close,rule);e.to_csv(out/f'events_2021_{name}.csv')
        (out/f'funnel_{name}.json').write_text(json.dumps(log['counts'],indent=2))
        for period,start,end in [('reused_development','2021-01-01','2021-07-01'),('reused_validation','2021-07-01','2021-10-01'),('reused_late_period','2021-10-01','2022-01-01')]:
            sub=e.loc[(e.index>=start)&(e.index<end)]
            rows.append({'variant':name,'period':period,'triggers':len(sub),'median_risk_thresholds':float(sub.risk_thresholds.median()),
                         'return':None,'SL_rate':None,'role':'PRICE_PATH_ONLY_NO_EXECUTABLE_QUOTES'})
        for month in range(1,13):months.append({'variant':name,'month':f'2021-{month:02d}','triggers':int((e.index.month==month).sum())})
        sample_events,sl=detect_structure(sample_bars.close,rule)
        sample_events.to_csv(out/f'sample_events_{name}.csv')
        for rr in [2.5,3.]:
            s=sample_events.copy();s['rr']=rr
            r=backtest(ticks,s,cfg);stats=metrics(r)[0]
            smokes.append({'variant':name,'RR':rr,'role':'REUSED_WEEK_SMOKE_NOT_OOS','detected_triggers':len(s),**stats})
            r['trades'].to_csv(out/f'sample_trades_{name}_{rr:g}R.csv',index=False)
        if name=='dc3_baseline':
            for stress_name,sc in [('spread1.5_slip2',replace(cfg,spread_multiplier=1.5,slippage=cfg.slippage*2)),('spread2_slip3',replace(cfg,spread_multiplier=2.,slippage=cfg.slippage*3))]:
                smokes.append({'variant':stress_name,'RR':2.5,'role':'BASELINE_COST_STRESS_SMOKE',
                               'detected_triggers':len(sample_events),**metrics(backtest(ticks,sample_events,sc))[0]})
            # Plot price-path extrema and causal confirmation (no candle bodies/wicks).
            import matplotlib.pyplot as plt
            for k,(time,ev) in enumerate(sample_events.head(3).iterrows()):
                x=sample_bars.close.loc[ev.a_time-pd.Timedelta(minutes=5):time+pd.Timedelta(minutes=10)]
                fig,ax=plt.subplots(figsize=(11,4));ax.plot(x.index,x.values)
                for label in ['a','b','c','d']:
                    ax.scatter(ev[label+'_time'],ev[label]);ax.annotate(label.upper(),(ev[label+'_time'],ev[label]))
                ax.axvline(time,color='black',ls='--',label='D confirmation/trigger; entry later quote')
                ax.axhline(ev.stop_price,color='red',ls=':',label='absolute stop');ax.axhline(ev.b,color='orange',ls='--',label='B')
                ax.legend();ax.set_title('Actual GBPUSD close-price path — not screenshot reproduction');fig.tight_layout()
                p=out/f'path_{k+1}.png';fig.savefig(p);plt.close(fig);figures.append(p)
    tables=pd.DataFrame(rows);monthly=pd.DataFrame(months);sample=pd.DataFrame(smokes)
    tables.to_csv(out/'2021_structure.csv',index=False);monthly.to_csv(out/'2021_monthly_opportunities.csv',index=False);sample.to_csv(out/'sample_financial_smoke.csv',index=False)
    summary={'status':'PRICE_PATH_IMPLEMENTED_NOT_PROFITABILITY_VERIFIED','2021_data':audit,'sample_quotes':tick_audit,
             'protocol_sha256':protocol['sha256'],'stable_system':False,'long_term_tick_backtest_completed':False,
             'baseline':'dc3_baseline; chosen before outcome comparison, NOT best performer',
             'candlestick_shape_removed':True,'time_sampling_removed':False,'price_series':'closed M1 close (sample: BID close)',
             'caveats':['Reuses previously inspected 2021 and 2016 datasets; no claim of untouched holdout.',
                        'Directional-change extrema occur before confirmation; signals dated confirmation ONLY. No repaint entries.',
                        'Prior EWMA absolute price changes; threshold frozen per leg. Not sampling-frequency invariant.',
                        'No multi-year executable quotes available; no cross-index transfer proven.',
                        'More signals are not evidence of higher accuracy. Small real-quote sample can lose money.',
                        'Relaxed retracement 65% is explicitly a departure from PDF, tested separately.',
                        'ML not fitted on tiny reused week; no claim of eliminating only losing trades.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import base64
    body='<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;margin:24px}table{font-size:12px}pre{white-space:pre-wrap}</style><h1>Price-path pivot research — NOT a stable system</h1><h2>2021 structure only</h2>'+tables.to_html(index=False)+'<h2>Real bid/ask week — reused smoke only</h2>'+sample.to_html(index=False)
    for p in figures:body+='<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
    body+='<pre>'+json.dumps(summary,indent=2)+'</pre>'; (out/'report.html').write_text(body)
    print(sample[['variant','RR','trades','total_return','max_drawdown']].to_string(index=False))

if __name__=='__main__':main()
