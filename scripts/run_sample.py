"""python -m scripts.run_sample"""
import json
from pathlib import Path
import pandas as pd
from hippo_lab.data import download_sample, make_bars
from hippo_lab.engine import Execution, backtest
from hippo_lab.research import research_run, CANDIDATES, signals_for, fit_candidate
from hippo_lab.report import metrics, write_report


def main():
    ticks,audit=download_sample(); bars=make_bars(ticks); cfg=Execution()
    run=research_run(ticks,bars,cfg,mode='sample')
    rows=[]
    for c in CANDIDATES:
        # Diagnostic replay only, never used to override validation rejection.
        model=fit_candidate(c,bars,ticks.index[0],pd.Timestamp('2016-04-07',tz='America/New_York').tz_convert('UTC'))
        sig=signals_for(c,bars,ticks.index[0],ticks.index[-1],model)
        r=backtest(ticks,sig,cfg); stats=metrics(r)[0]
        rows.append({'candidate':c['name'],'role':'SMOKE ONLY — not OOS', **stats})
    diagnostics=pd.DataFrame(rows)
    summary=write_report(run,audit,diagnostics=diagnostics)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/sample_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    run['selection'].to_csv('reports/sample_validation.csv',index=False)
    diagnostics.to_csv('reports/sample_diagnostics.csv',index=False)
    print(json.dumps(summary,indent=2,ensure_ascii=False)); print(diagnostics.to_string(index=False))

if __name__=='__main__':main()
