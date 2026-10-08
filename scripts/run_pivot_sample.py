from pathlib import Path
from hippo_lab.data import download_sample
from hippo_lab.engine import Execution
from hippo_lab.pivot_experiment import run_pivot_experiment
import json,shutil

def main():
    ticks,audit=download_sample()
    report,table,results=run_pivot_experiment(ticks,audit,Execution(),mode='sample')
    print(table.to_string(index=False)); print('ML:',report['ML']);print('frozen',report['frozen_sha256'])
    dest=Path('reports/pivot');dest.mkdir(parents=True,exist_ok=True)
    src=Path('artifacts/pivot')
    for name in ['summary.json','test_comparison.csv','sample_full_week.csv','funnel.csv','report.html','frozen_before_test.json','development_rule_ablations.csv','smoke_trades_wick_2.5R.csv','smoke_trades_wick_3R.csv']:
        shutil.copy2(src/name,dest/name)

if __name__=='__main__':main()
