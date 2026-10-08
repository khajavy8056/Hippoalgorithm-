"""Auditable metrics; partial months never count toward the monthly objective."""
from pathlib import Path
import html, json, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .research import PROTOCOL


def metrics(result):
    e=result['equity']; cash=result['config']['initial_cash']; t=result['trades']
    # Avoid duplicate timestamp reindexing but preserve full quote-level DD from engine.
    e=e.groupby(level=0).last()
    daily=e.resample('1D').last().ffill()
    daily_returns=daily.pct_change(fill_method=None); daily_returns.iloc[0]=daily.iloc[0]/cash-1
    first=e.index[0]; last=e.index[-1]
    month_end=e.resample('ME').last()
    rows=[]; prev=cash
    for timestamp,value in month_end.items():
        start=timestamp.normalize().replace(day=1)
        next_start=start+pd.DateOffset(months=1)
        # Data must bracket a whole UTC month. Missing weekends are allowed up to 3 days.
        complete=first<=start+pd.Timedelta(days=3) and last>=next_start-pd.Timedelta(days=3)
        rows.append({'month':str(start.date())[:7], 'return':value/prev-1, 'complete':complete})
        prev=value
    monthly=pd.DataFrame(rows)
    pnl=t.pnl if len(t) else pd.Series(dtype=float)
    wins=pnl[pnl>0].sum(); losses=-pnl[pnl<0].sum()
    stats={'total_return':float(e.iloc[-1]/cash-1), 'max_drawdown':result['max_drawdown'],
           'trades':len(t),'win_rate':float((pnl>0).mean()) if len(t) else None,
           'profit_factor':float(wins/losses) if losses>0 else None,
           'net_pnl':float(e.iloc[-1]-cash), 'mean_trade_pnl':float(pnl.mean()) if len(t) else None,
           'gap_affected_trades':int(t.quote_gap_over_60s.sum()) if len(t) else 0,
           'gap_affected_pnl':float(t.loc[t.quote_gap_over_60s,'pnl'].sum()) if len(t) else 0.,
           'worst_realized_R':float(t.realized_R.min()) if len(t) else None,
           'complete_months':int(monthly.complete.sum()),'drawdown_breaker_triggered':result['halted'],
           'breaker_overshoot':max(0.,result['max_drawdown']-result['config']['drawdown_breaker']),
           'submitted_signals':result['submitted_signals'], 'stale_or_outside_signals':result['stale_or_outside_signals']}
    full=monthly.loc[monthly.complete,'return']
    stats.update(mean_complete_month_return=float(full.mean()) if len(full) else None,
                 fraction_months_ge_3pct=float((full>=.03).mean()) if len(full) else None,
                 worst_month=float(full.min()) if len(full) else None)
    # Bootstrap contiguous blocks only with enough days, not a week-long smoke test.
    bootstrap=None
    if len(daily_returns)>=180:
        x=daily_returns.to_numpy(); rng=np.random.default_rng(8056); samples=[]; dds=[]
        for _ in range(1000):
            starts=rng.integers(0,len(x)-20+1,size=int(np.ceil(len(x)/20)))
            r=np.concatenate([x[j:j+20] for j in starts])[:len(x)]
            curve=np.r_[1.,np.cumprod(1+r)]
            samples.append(curve[-1]**(30/len(r))-1)
            dds.append(np.max(1-curve/np.maximum.accumulate(curve)))
        bootstrap={'monthly_equivalent_mean_ci95':np.quantile(samples,[.025,.975]).tolist(),
                   'dd_p95':float(np.quantile(dds,.95)),
                   'note':'Conditional 20-calendar-day block resampling; not a guarantee or selection-bias correction.'}
    return stats,monthly,daily_returns,bootstrap


def write_report(run,audit,folder='artifacts',diagnostics=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    result=run['final']; stats,monthly,daily,bootstrap=metrics(result)
    no_selection=run['frozen']['candidate']=='CASH'
    enough=run['mode']!='sample' and stats['trades']>=30 and stats['complete_months']>=5
    status='NO_SELECTION' if no_selection and run['mode']!='sample' else ('INSUFFICIENT_EVIDENCE' if not enough else 'EVALUATED_NOT_GUARANTEED')
    # Walk-forward and final holdout are separate accounts, not an invented stitched equity curve.
    wf_stats=None
    if run['walk_forward'] is not None:
        wf_stats=metrics(run['walk_forward'])[0]
    stresses={k:metrics(v)[0] for k,v in run['stress'].items()}
    forward_folds=max(0,run['selection'].fold.nunique()-1) if run['mode']!='sample' else 0
    sufficient_folds=forward_folds>=PROTOCOL['minimum_forward_folds']
    point_met=bool(enough and not result['halted'] and stats['mean_complete_month_return']>=.03 and stats['max_drawdown']<.10 and all(s['total_return']>0 and not s['drawdown_breaker_triggered'] for s in stresses.values()))
    summary={'status':status, 'point_estimate_met_unconfirmed':point_met,
             'forward_folds':forward_folds, 'sufficient_forward_folds':sufficient_folds,
             'long_term_target_verified':False, 'input_sufficient_for_full_protocol':run['mode']!='sample',
             'no_selection':no_selection, 'holdout_evidence_sufficient_for_initial_review':enough,
             'monthly_goal_definition':'Mean net complete calendar-month return >=3%; not a promise of 3% every month. Long-term target requires >=24 OOS months.','objective':'3–4% monthly, maximum equity drawdown <10%; not guaranteed',
             'data_audit':audit,'frozen_selection':run['frozen'],'final_holdout':stats,
             'walk_forward_separate_account':wf_stats,'stress_same_frozen_signals':stresses,
             'bootstrap':bootstrap, 'limitations':[
                 'Public mirror quotes not independently broker-certified; no exchange order book or market impact.',
                 'Quote-currency account only (GBPUSD: USD). Other pairs require their quote-currency account; no cross-FX conversion.',
                 'Single position, intraday NY liquidation; no swaps; observed quote gaps can overshoot risk controls.',
                 'Best-of-six selection is biased; holdout is not used for tuning. Revisiting after results voids its untouched status.',
                 'One week cannot establish profitability or transfer across markets; >=24 input months is an exploratory run, not 24 OOS months.',
                 'Validation reject -> no deployment, not evidence that cash is superior.']}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    run['selection'].to_csv(folder/'validation_all_candidates.csv',index=False)
    result['trades'].to_csv(folder/'holdout_trades.csv',index=False)
    monthly.to_csv(folder/'holdout_months.csv',index=False)
    result['equity'].resample('1min').last().to_csv(folder/'equity_1min.csv')
    # Full tick equity remains in memory; optional parquet audit, not committed.
    result['equity'].to_frame().to_parquet(folder/'equity_every_tick.parquet')
    if diagnostics is not None: diagnostics.to_csv(folder/'smoke_unselected_candidates.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    plot_e=result['equity'].groupby(level=0).last()
    axes[0].plot(plot_e.index,plot_e.values); axes[0].set_ylabel('Liquidation equity')
    peak=plot_e.cummax().clip(lower=result['config']['initial_cash'])
    axes[1].plot(plot_e.index,100*(1-plot_e/peak)); axes[1].set_ylabel('Drawdown %')
    fig.suptitle(status+' — frozen holdout (CASH may reflect inadequate evidence)')
    fig.tight_layout(); fig.savefig(folder/'equity.png'); plt.close(fig)
    def table(df): return df.to_html(index=False,escape=True,float_format=lambda x:f'{x:.6f}')
    body=f'''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><title>HIPO audit</title>
    <style>body{{font-family:system-ui;max-width:1100px;margin:32px auto;padding:20px;background:#f5f7fb;color:#17233a}}table{{direction:ltr;border-collapse:collapse;width:100%;background:white}}td,th{{padding:8px;border:1px solid #ccd}}pre{{direction:ltr;text-align:left;white-space:pre-wrap}}img{{width:100%}}</style>
    <h1>گزارش پژوهش تیک‌به‌تیک HIPO</h1><h2>{status}</h2>
    <p>این گزارش تضمین سود نیست. داده کوتاه فقط آزمون مهندسی است؛ عدم معامله، تأیید استراتژی نیست.</p>
    <img src="data:image/png;base64,{base64.b64encode((folder/'equity.png').read_bytes()).decode()}"><h2>تمام آزمایش‌های اعتبارسنجی</h2>{table(run['selection'])}
    <h2>ماه‌ها (ناقص‌ها خارج از هدف)</h2>{table(monthly)}
    <h2>اجرای تشخیصی فرضیه‌های از پیش تعیین‌شده — نه انتخاب و نه اثبات سود</h2>{table(diagnostics) if diagnostics is not None else ''}
    <h2>فرض‌ها، داده و تنش هزینه</h2><pre>{html.escape(json.dumps(summary,indent=2,ensure_ascii=False))}</pre></html>'''
    (folder/'report.html').write_text(body,encoding='utf-8')
    return summary
