"""Long-history quote validation and year/quarter financial audit on supplied real ticks."""
from pathlib import Path
import json
import pandas as pd
from .pivot_experiment import run_pivot_experiment
from .engine import Execution


def long_quote_gate(ticks, minimum_days=365):
    if not {'bid','ask'}<=set(ticks):raise ValueError('Real bid AND ask required; OHLC cannot satisfy long tick backtest')
    if not isinstance(ticks.index,pd.DatetimeIndex) or ticks.index.tz is None or not ticks.index.is_monotonic_increasing:
        raise ValueError('Sorted timezone-aware real quotes required')
    span=(ticks.index[-1]-ticks.index[0]).total_seconds()/86400
    # Allow weekend start/end in a nominal calendar year; cannot certify broker provenance.
    if span<minimum_days-7:raise ValueError(f'Need approximately {minimum_days} days real quotes; have {span:.2f}')
    dates=pd.date_range(ticks.index[0].normalize(),ticks.index[-1].normalize(),freq='B',tz='UTC')
    missing=dates.difference(ticks.index.normalize().unique())
    fraction=len(missing)/max(1,len(dates))
    if fraction>.03:raise ValueError(f'Missing {len(missing)} weekday dates ({fraction:.1%}); repair coverage first')
    return {'span_days':span,'missing_weekdays':[str(d.date()) for d in missing],
            'missing_weekday_fraction':fraction,'coverage_gate_passed':True,
            'warning':'Gate checks coverage only, not quote authenticity or suitability for execution.'}


def run_long_pivot(ticks,audit,cfg=Execution(),folder='artifacts/pivot_long',minimum_days=365):
    coverage=long_quote_gate(ticks,minimum_days)
    p=Path(folder);p.mkdir(parents=True,exist_ok=True)
    audit={**audit,'long_coverage':coverage}
    # Frozen 60/20/20 initial experiment; no repeated changes to its final 20%.
    report,comparison,results=run_pivot_experiment(ticks,audit,cfg,folder=folder,mode='research')
    # Calendar-year and quarter net equity changes (NOT independent reinitialised accounts).
    rows=[]
    for name,result in results.items():
        eq=result['equity'].groupby(level=0).last()
        for freq in ['YE','QE']:
            marks=eq.resample(freq).last();previous=cfg.initial_cash
            for time,value in marks.items():
                rows.append({'variant':name,'period_end':str(time),'frequency':freq,
                             'net_equity_return':float(value/previous-1),
                             'partial_boundary_period_possible':True,
                             'note':'Sequential OOS account, not a fresh independent year/quarter backtest.'})
                previous=value
    pd.DataFrame(rows).to_csv(p/'OOS_year_quarter_equity.csv',index=False)
    report['long_quote_coverage']=coverage
    report['stability_verified']=False
    report['required_next_stage']='Multiple chronological walk-forward folds and a fresh untouched future holdout; no stability claim from one split.'
    (p/'long_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str).replace('NaN','null'))
    return report,comparison,results
