"""One-year genuine M1 structural audit. NO financial backtest or invented quotes."""
from pathlib import Path
from dataclasses import replace,asdict
import hashlib,json
import pandas as pd
from hippo_lab.pivot import detect_pivots,PivotRules,validate_bars

COMMIT='ef259913eb77119b95a56e99a1aff0146de249fd'

def load_year(path='data/long/eurusd_m1.csv'):
    raw=pd.read_csv(path)
    # Source timestamps are naive and undocumented. UTC is ONLY a coordinate assumption.
    idx=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:,0])).tz_localize('UTC')+pd.Timedelta(minutes=1)
    bars=raw.iloc[:,2:6].copy();bars.columns=['open','high','low','close'];bars.index=idx
    duplicate=int(bars.index.duplicated().sum())
    if duplicate:raise ValueError('Duplicate M1 timestamps: inspect source before audit')
    bars=validate_bars(bars.sort_index())
    year=bars.loc[(bars.index>=pd.Timestamp('2021-01-01',tz='UTC'))&(bars.index<pd.Timestamp('2022-01-01',tz='UTC'))]
    weekdays=pd.date_range('2021-01-01','2021-12-31',freq='B',tz='UTC')
    observed=year.index.normalize().unique()
    audit={'source':'https://github.com/hftradingstrategies/fxdatasets','commit':COMMIT,
           'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'pair':'EURUSD',
           'type':'M1 OHLC ONLY; NOT tick bid/ask','source_timezone':'undocumented; UTC coordinate assumption, not certified',
           'timestamp_convention':'assumed bar open, +1min to close; not source-certified',
           'year':2021,'calendar_year_complete':False,'calendar_months_touched':12,'rows':len(year),'start':str(year.index[0]),'end':str(year.index[-1]),
           'duplicate_timestamps':duplicate,'missing_weekday_dates':[str(t.date()) for t in weekdays.difference(observed)],
           'gaps_over_3min':int((year.index.to_series().diff()>pd.Timedelta(minutes=3)).sum()),
           'flat_bar_fraction':float(((year.high-year.low)==0).mean()),
           'broker_certified':False,'financial_backtest_permitted':False}
    return year,audit


def main():
    out=Path('reports/pivot_year');out.mkdir(parents=True,exist_ok=True)
    bars,audit=load_year();rows=[];monthly=[]
    # Fixed intervals: development Jan-Jun, validation Jul-Sep, holdout Oct-Dec.
    splits=[('development','2021-01-01','2021-07-01'),('validation','2021-07-01','2021-10-01'),('holdout','2021-10-01','2022-01-01')]
    variants=[('baseline',PivotRules()),('close_boundary',replace(PivotRules(),boundary='close')),
              ('body_050',replace(PivotRules(),signal_body_ratio=.50)),
              ('body_070',replace(PivotRules(),signal_body_ratio=.70)),
              ('no_FL',replace(PivotRules(),allow_fl=False)),
              ('time_divergence',replace(PivotRules(),require_time_divergence=True)),
              ('impulse_efficiency_045',replace(PivotRules(),ab_efficiency=.45)),
              ('trigger_body_035',replace(PivotRules(),trigger_body_atr=.35))]
    frozen={'type':'STRUCTURAL_AUDIT_ONLY','variants':{n:asdict(r) for n,r in variants},'splits':splits,
            'no_return_or_SL_computation':True,'no_selection_on_count_alone':True,'data':audit}
    frozen['sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    old_protocol=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else None
    cache_valid=old_protocol is not None and old_protocol.get('variants')==frozen['variants'] and old_protocol.get('data',{}).get('sha256')==audit['sha256']
    (out/'protocol.json').write_text(json.dumps(frozen,indent=2))
    for name,rule in variants:
        print('Audit',name,flush=True)
        # detect once causally, then split events; allows historical context like a live strategy.
        cache=out/f'events_{name}.csv'
        if cache.exists() and cache_valid:
            events=pd.read_csv(cache,index_col=0,parse_dates=True)
            events.index=pd.DatetimeIndex(events.index)
            log={'counts':json.loads((out/f'funnel_{name}.json').read_text())}
        else:
            events,log=detect_pivots(bars,rule)
        events.to_csv(out/f'events_{name}.csv')
        (out/f'funnel_{name}.json').write_text(json.dumps(log['counts'],indent=2))
        for stage,start,end in splits:
            b=bars.loc[(bars.index>=start)&(bars.index<end)]
            e=events.loc[(events.index>=start)&(events.index<end)]
            rows.append({'variant':name,'period':stage,'m1_bars':len(b),'triggers':len(e),
                         'buys':int((e.side==1).sum()),'sells':int((e.side==-1).sum()),
                         'median_risk_atr':float(e.risk_atr.median()) if len(e) else None,
                         'median_cd_ab':float(e.cd_ab_ratio.median()) if len(e) else None,
                         'return':None,'stop_rate':None,'status':'NO_EXECUTABLE_QUOTES'})
        for month in range(1,13):
            m=events.loc[events.index.month==month]
            monthly.append({'variant':name,'month':f'2021-{month:02d}','triggers':len(m)})
    table=pd.DataFrame(rows);table.to_csv(out/'structural_comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'monthly_opportunities.csv',index=False)
    summary={'status':'2021_PARTIAL_YEAR_STRUCTURE_TESTED_FINANCIAL_NOT_TESTED','data_audit':audit,
             'protocol_sha256':frozen['sha256'],'profitability_verified':False,'stable_system_verified':False,
             'coverage_warning':'First available 2021 bar is Jan 11; six weekday dates absent. Not a complete one-year quote backtest.',
             'warning':'More triggers or narrower structural stops do not imply better precision/profitability. No variant chosen.',
             'tested':'2021 causal M1 detection and eight predeclared structural variants',
             'not_tested':'2021 bid/ask tick execution, returns, stop frequency, ML winner preservation, two years, DAX',
             'comparison':table.where(pd.notna(table),None).to_dict('records')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2).replace('NaN','null'))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>2021 Pivot structural audit — NOT financial backtest</h1><p>Actual OHLC; no executable quotes. No performance or stability claim.</p>'+table.to_html(index=False)+'<pre>'+json.dumps(audit,indent=2)+'</pre>')
    print(table.to_string(index=False))

if __name__=='__main__':main()
