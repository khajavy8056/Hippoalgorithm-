"""Experiments restricted to pivot settlement, never a search over unrelated strategies."""
from dataclasses import replace, asdict
from pathlib import Path
import hashlib, json, html, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .pivot import PivotRules, FEATURE_COLUMNS, detect_pivots, pivot_bars
from .engine import Execution, backtest
from .report import metrics

PROTOCOL = {'variants':['wick_2.5R','wick_3R','close_2.5R','close_3R'],
            'baseline':'wick_2.5R', 'chronology':'60% train / 20% validation / 20% test of elapsed UTC time',
            'thresholds':[.35,.40,.45,.50], 'winner_retention_min':.95,
            'positive_pnl_retention_min':.95, 'losers_removed_min':.10,
            'minimum_train_labels':100, 'minimum_validation_winners':30,
            'minimum_validation_losers':30, 'embargo_hours':24,
            'seed':8056, 'minimum_input_months_for_ml':12,
            'target':'Mean net complete monthly return >=3%, equity DD <10%; >=24 OOS months for long-term review',
            'sample_status':'ENGINEERING_ONLY', 'no_passing_filter':'bypass, preserve baseline signals',
            'no_variant_selection_on_test':True}


def paired_retention(labels, keep):
    labels=labels.copy(); keep=np.asarray(keep,dtype=bool)
    win=labels.pnl>0; loss=labels.pnl<0
    return {'winners':int(win.sum()),'losers':int(loss.sum()),
        'kept_winners':int((win&keep).sum()),'removed_winners':int((win&~keep).sum()),
        'kept_losers':int((loss&keep).sum()),'removed_losers':int((loss&~keep).sum()),
        'winner_retention':float(keep[win].mean()) if win.any() else None,
        'positive_pnl_retention':float(labels.loc[win&keep,'pnl'].sum()/labels.loc[win,'pnl'].sum()) if win.any() else None,
        'losers_removed_fraction':float((~keep)[loss].mean()) if loss.any() else None,
        'removed_positive_pnl':float(labels.loc[win&~keep,'pnl'].sum()),
        'avoided_negative_pnl':float(-labels.loc[loss&~keep,'pnl'].sum()),
        'isolated_net_pnl_delta':float(-labels.loc[~keep,'pnl'].sum()),
        'note':'Paired ISOLATED trades. Portfolio effect differs: removing a trade can allow a later entry.'}


def isolated_labels(ticks,events,cfg):
    """Labels use actual quote replay and absolute stop. Not vectorized candle approximations."""
    records=[]
    for j,(time,event) in enumerate(events.iterrows()):
        end=time+pd.Timedelta(hours=cfg.max_hold_hours)+pd.Timedelta(minutes=5)
        lo=ticks.index.searchsorted(time,side='left'); hi=ticks.index.searchsorted(end,side='right')
        window=ticks.iloc[lo:hi]
        if len(window)<2:continue
        r=backtest(window,events.loc[[time]],cfg)
        if len(r['trades'])!=1:continue  # session/spread/risk/latency rejected => no training label
        t=r['trades'].iloc[0]
        if int(t.reason)==6:continue # right-censored window/end-of-file is NOT a resolved training outcome
        rec={f:float(event[f]) for f in FEATURE_COLUMNS}
        rec.update(time=time,label_exit=t.exit_time,pnl=float(t.pnl),R=float(t.realized_R),
                   side=int(event.side),reason=int(t.reason),setup_id=event.setup_id)
        records.append(rec)
    if records:return pd.DataFrame(records).set_index('time').sort_index()
    return pd.DataFrame(columns=[*FEATURE_COLUMNS,'label_exit','pnl','R','side','reason','setup_id'],index=pd.DatetimeIndex([],tz='UTC',name='time'))


def nonoverlapping(labels):
    # Independent exposure windows for fitting; purge is on label end, not rows.
    indices=[];last=None
    for i,(time,row) in enumerate(labels.iterrows()):
        if last is None or time>last:
            indices.append(i);last=row.label_exit
    return labels.iloc[indices]


def fit_filter(labels,events,train_end,val_end,enough_history):
    info={'enabled':False,'reason':'INSUFFICIENT_HISTORY','threshold':None,'training_labels':0,'validation_labels':0}
    if not enough_history:
        info['available_development_labels']=len(labels)
        return None,None,pd.DataFrame(),info
    train=labels.loc[(labels.index<train_end)&(labels.label_exit<train_end-pd.Timedelta(hours=PROTOCOL['embargo_hours']))]
    train=nonoverlapping(train).dropna(subset=FEATURE_COLUMNS)
    val=labels.loc[(labels.index>=train_end)&(labels.index<val_end)&(labels.label_exit<val_end-pd.Timedelta(hours=PROTOCOL['embargo_hours']))].dropna(subset=FEATURE_COLUMNS)
    info.update(training_labels=len(train),validation_labels=len(val))
    if len(train)<PROTOCOL['minimum_train_labels'] or (train.pnl>0).nunique()<2:
        info['reason']='INSUFFICIENT_PURGED_TRAINING_LABELS';return None,None,pd.DataFrame(),info
    if int((val.pnl>0).sum())<PROTOCOL['minimum_validation_winners'] or int((val.pnl<0).sum())<PROTOCOL['minimum_validation_losers']:
        info['reason']='INSUFFICIENT_VALIDATION_WINNERS_AND_LOSERS';return None,None,pd.DataFrame(),info
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=8056))
    model.fit(train[FEATURE_COLUMNS].astype(float),(train.pnl>0).astype(int))
    probs=model.predict_proba(val[FEATURE_COLUMNS].astype(float))[:,1]
    rows=[];best=None;best_gain=0.
    for threshold in PROTOCOL['thresholds']:
        m=paired_retention(val,probs>=threshold)
        passes=m['winner_retention']>=PROTOCOL['winner_retention_min'] and m['positive_pnl_retention']>=PROTOCOL['positive_pnl_retention_min'] and m['losers_removed_fraction']>=PROTOCOL['losers_removed_min'] and m['isolated_net_pnl_delta']>0
        rows.append({'threshold':threshold,'eligible':passes,**m})
        if passes and m['isolated_net_pnl_delta']>best_gain:best=threshold;best_gain=m['isolated_net_pnl_delta']
    if best is None:
        info['reason']='NO_THRESHOLD_PRESERVES_VALIDATION_WINNERS';return None,None,pd.DataFrame(rows),info
    # NO refit on validation: keep scaler/coefficients used to choose threshold.
    info.update(enabled=True,reason='VALIDATION_ACCEPTED_NOT_GUARANTEED',threshold=best,
                coefficients=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist(),
                scaler_mean=model[0].mean_.tolist(),scaler_scale=model[0].scale_.tolist())
    return model,best,pd.DataFrame(rows),info


def plot_setup(bars,event,time,path):
    start=event.a_time-pd.Timedelta(minutes=10);end=time+pd.Timedelta(minutes=20)
    b=bars.loc[start:end]; fig,ax=plt.subplots(figsize=(12,4));axis=np.arange(len(b))
    up=b.close>=b.open
    for j,(_,row) in enumerate(b.iterrows()):
        color='#138f70' if up.iloc[j] else '#c23c4b'
        ax.vlines(j,row.low,row.high,color=color)
        ax.bar(j,abs(row.close-row.open),bottom=min(row.close,row.open),width=.65,color=color)
    positions=[b.index.searchsorted(t) for t in [event.a_time,event.b_time,event.c_time,event.signal_time]]
    prices=[event.a,event.b,event.c,event.d]
    ax.plot(positions,prices,'o-',color='#1764b4')
    for j,p,label in zip(positions,prices,['A','B','C','D (max to trigger)']):ax.annotate(label,(j,p))
    ax.axhline(event.signal_level,color='orange',ls='--',label='signal low/high')
    ax.axhline(event.stop_price,color='red',ls=':',label='absolute stop')
    ax.axvline(b.index.searchsorted(time),color='black',ls='--',label='trigger CLOSE (entry later tick)')
    ax.set_title(f'Automatic GBPUSD M1 pivot — {time}; NOT a DAX screenshot reproduction')
    ax.set_xticks(axis[::max(1,len(b)//10)]);ax.set_xticklabels([str(t)[11:16] for t in b.index[::max(1,len(b)//10)]])
    ax.legend();fig.tight_layout();fig.savefig(path);plt.close(fig)


def run_pivot_experiment(ticks,audit,cfg=Execution(),folder='artifacts/pivot',mode='sample',bars=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    bars=pivot_bars(ticks) if bars is None else bars
    start,end=ticks.index[0],ticks.index[-1]
    train_end=start+(end-start)*.60;test_start=start+(end-start)*.80
    test_ticks=ticks.loc[ticks.index>=test_start]
    if len(test_ticks)<2:raise ValueError('Test has insufficient quotes')
    input_days=(end-start).total_seconds()/86400
    histories={};all_events={};counts=[]
    for boundary in ['wick','close']:
        e,log=detect_pivots(bars,replace(PivotRules(),boundary=boundary));all_events[boundary]=e
        histories[boundary]=log
        (folder/f'trace_{boundary}.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
        e.to_csv(folder/f'all_events_{boundary}.csv')
        counts.append({'boundary':boundary,**log['counts']})
    # Fixed baseline; four variants are sensitivity diagnostics, NOT best-of-four selection.
    base_events=all_events['wick'].copy()
    # Drop pending triggers whose setup started before the test boundary from training LABELS
    # only labels are purged; causal setup context may span split just as live trading does.
    dev_events=base_events.loc[base_events.index<test_start]
    labels=isolated_labels(ticks.loc[ticks.index<test_start],dev_events,cfg)
    model,threshold,validation,filter_info=fit_filter(labels,dev_events,train_end,test_start,
                                                    enough_history=(mode!='sample' and input_days>=365))
    frozen={'protocol':PROTOCOL,'baseline_rules':asdict(PivotRules()),'execution':asdict(cfg),
            'test_start':str(test_start),'train_end':str(train_end),'ml':filter_info,
            'data_sha256':audit.get('sha256',audit.get('manifest'))}
    frozen['sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True,default=str).encode()).hexdigest()
    (folder/'frozen_before_test.json').write_text(json.dumps(frozen,indent=2,default=str))
    rows=[];results={};smoke=[]
    for boundary in ['wick','close']:
        for rr in [2.5,3.]:
            name=f'{boundary}_{rr:g}R';events=all_events[boundary].copy();events['rr']=rr
            s=events.loc[events.index>=test_start]
            result=backtest(test_ticks,s,cfg);stats,monthly,daily,bootstrap=metrics(result)
            rows.append({'variant':name,'role':'OOS sensitivity; NOT selected',**stats});results[name]=result
            result['trades'].to_csv(folder/f'test_trades_{name}.csv',index=False)
            monthly.to_csv(folder/f'test_months_{name}.csv',index=False)
            result['equity'].to_frame().to_parquet(folder/f'test_tick_equity_{name}.parquet')
            if mode=='sample':
                r=backtest(ticks,events,cfg)
                r['trades'].to_csv(folder/f'smoke_trades_{name}.csv',index=False)
                smoke.append({'variant':name,'role':'ALL WEEK SMOKE — NOT OOS',**metrics(r)[0]})
    test_events=base_events.loc[base_events.index>=test_start]
    keep=np.ones(len(test_events),dtype=bool)
    if model is not None and len(test_events):
        keep=model.predict_proba(test_events[FEATURE_COLUMNS].astype(float))[:,1]>=threshold
    filtered=backtest(test_ticks,test_events.loc[keep],cfg);results['wick_2.5R_ML']=filtered
    rows.append({'variant':'wick_2.5R_ML','role':'OOS frozen filter or bypass',**metrics(filtered)[0]})
    filtered['trades'].to_csv(folder/'test_trades_ML.csv',index=False)
    # Isolated OOS labels ONLY now, after frozen model/threshold. Never used to repair filtering.
    test_events.assign(ML_keep=keep).to_csv(folder/'test_signal_decisions.csv')
    test_labels=isolated_labels(test_ticks,test_events,cfg)
    label_keep=np.ones(len(test_labels),dtype=bool)
    if model is not None and len(test_labels):label_keep=model.predict_proba(test_labels[FEATURE_COLUMNS].astype(float))[:,1]>=threshold
    preservation=paired_retention(test_labels,label_keep)
    if len(test_labels):
        test_labels['kept']=label_keep
        test_labels.to_csv(folder/'paired_OOS_winners_losers.csv')
    stress={}
    for name,stress_cfg in [('spread1.5_slip2',replace(cfg,spread_multiplier=1.5,slippage=cfg.slippage*2)),
                            ('spread2_slip3_latency1s',replace(cfg,spread_multiplier=2.,slippage=cfg.slippage*3,latency_ms=1000))]:
        stress[name]=metrics(backtest(test_ticks,test_events.loc[keep],stress_cfg))[0]
    filter_months=metrics(filtered)[1]; filter_bootstrap=metrics(filtered)[3]
    metrics_df=pd.DataFrame(rows);metrics_df.to_csv(folder/'test_comparison.csv',index=False)
    validation.to_csv(folder/'ML_validation_thresholds.csv',index=False)
    pd.DataFrame(counts).to_csv(folder/'funnel.csv',index=False)
    pd.DataFrame(smoke).to_csv(folder/'sample_full_week.csv',index=False)
    # Predeclared rule ablations on DEVELOPMENT only, never override baseline on test.
    ablations=[]
    for name,rule in [
        ('body_ratio_0.50',replace(PivotRules(),signal_body_ratio=.50)),
        ('body_ratio_0.70',replace(PivotRules(),signal_body_ratio=.70)),
        ('no_FL',replace(PivotRules(),allow_fl=False)),
        ('require_BC_time_divergence',replace(PivotRules(),require_time_divergence=True)),
    ]:
        dev_bars=bars.loc[bars.index<test_start]
        dev_s,dev_log=detect_pivots(dev_bars,rule)
        dev_r=backtest(ticks.loc[ticks.index<test_start],dev_s,cfg)
        ablations.append({'ablation':name,'role':'DEVELOPMENT ONLY — NOT SELECTED',
                          'triggers':len(dev_s),**metrics(dev_r)[0]})
    pd.DataFrame(ablations).to_csv(folder/'development_rule_ablations.csv',index=False)
    smoke_stress={}
    if mode=='sample':
        for name,stress_cfg in [('spread1.5_slip2',replace(cfg,spread_multiplier=1.5,slippage=cfg.slippage*2)),
                                ('spread2_slip3_latency1s',replace(cfg,spread_multiplier=2.,slippage=cfg.slippage*3,latency_ms=1000))]:
            smoke_stress[name]=metrics(backtest(ticks,base_events,stress_cfg))[0]
    # A time-divergence-only hypothesis reported on development only; not selected on OOS.
    td=dev_events.loc[dev_events.time_divergence.astype(float)==1] if len(dev_events) else dev_events
    dev_ticks=ticks.loc[ticks.index<test_start]
    td_stats=metrics(backtest(dev_ticks,td,cfg))[0]
    report={'status':'INSUFFICIENT_EVIDENCE' if mode=='sample' else 'OOS_RESEARCH_NOT_GUARANTEED',
        'pdf_rules_operational_not_confirmed':True,'exact_DAX_image_signal_match_verified':False,
        'long_term_monthly_target_verified':False,'mean_monthly_target_observed':False,
        'full_history_validation_complete':False,'mode':mode,'data_audit':audit,
        'development_rule_ablation_trials':len(ablations),'development_ablations':ablations,
        'sample_full_week_cost_stress_NOT_OOS':smoke_stress,'rule_variants_trial_count':4,'ML_threshold_trials':len(PROTOCOL['thresholds']),
        'closed_bid_bar_count':len(bars),'input_days':input_days,'splits':{'train_end':str(train_end),'test_start':str(test_start)},
        'frozen_sha256':frozen['sha256'],'ML':filter_info,'paired_test_preservation':preservation,
        'complete_filtered_test_months':int(filter_months.complete.sum()),
        'filtered_test_bootstrap':filter_bootstrap, 'stress_filtered_baseline':stress,'optional_time_divergence_development_only':td_stats,
        'oos_results':metrics_df.where(pd.notna(metrics_df),None).to_dict('records'),
        'limitations':['No matching DAX/FXCM raw candles/ticks or chart timezone supplied. Image audit is qualitative only.',
                      'Only GBPUSD real bid/ask mirror data executed here; not DAX profitability or image reproduction.',
                      'Numerical candle thresholds, impulse segmentation, B-position band and expiry are explicit operational assumptions.',
                      'C is provisionally updated then frozen at B breakout; BC duration is B exclusive through final C inclusive. Base after C logged separately.',
                      'Fixed TP 2.5/3R per user; screenshot name returns are NOT training labels or expected exits.',
                      'Validation preservation is not guaranteed out of sample. Removed winners and positive PnL must be reported.',
                      'Same-tick liquidity, market impact, swaps and cross-currency account conversion not modeled; risk breakers can gap.',
                      'OHLC is for signals ONLY. Prices and absolute stops executed with real bid/ask on strictly later quotes.',
                      'Short sample cannot fit ML or establish monthly target. Bypass does not prove an ML improvement.',
                      'Four OOS variants are sensitivity tests, not independent untouched proofs after human comparison.']}
    # Convert NaN to null for honest JSON.
    text=json.dumps(report,indent=2,ensure_ascii=False,default=str)
    text=text.replace('NaN','null');(folder/'summary.json').write_text(text)
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for name,r in results.items():
        eq=r['equity'].groupby(level=0).last().resample('1min').last().dropna()
        axes[0].plot(eq.index,eq,label=name)
        axes[1].plot(eq.index,100*(1-eq/eq.cummax().clip(lower=cfg.initial_cash)),label=name)
    axes[0].set_ylabel('Liquidation equity');axes[1].set_ylabel('DD %');axes[0].legend()
    fig.tight_layout();fig.savefig(folder/'comparison.png');plt.close(fig)
    illustrations=[]
    for k,(time,event) in enumerate(base_events.head(4).iterrows()):
        p=folder/f'auto_setup_{k+1}.png';plot_setup(bars,event,time,p);illustrations.append(p)
    image=lambda p:'<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
    body='<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><title>Pivot settlement audit</title><style>body{font-family:system-ui;max-width:1200px;margin:24px auto;padding:20px}table{direction:ltr;font-size:12px;border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}pre{direction:ltr;text-align:left;white-space:pre-wrap}</style>'
    body+='<h1>پیوت تسویه — گزارش پژوهش</h1><p>تطبیق عددی تصاویر DAX انجام نشده؛ این نمودارهای خودکار GBPUSD هستند. نتیجه کوتاه، تضمین سود ماهانه نیست.</p>'
    body+=image(folder/'comparison.png')+'<h2>حساسیت خارج از نمونه — بدون انتخاب برنده</h2>'+metrics_df.to_html(index=False)
    body+='<h2>آزمون مهندسی کل هفته — نه خارج از نمونه</h2>'+pd.DataFrame(smoke).to_html(index=False)
    body+='<h2>تغییرات همان منطق — فقط بخش توسعه، بدون انتخاب روی تست</h2>'+pd.DataFrame(ablations).to_html(index=False)
    body+='<h2>حفظ سودها و حذف خطاها</h2><pre>'+html.escape(json.dumps(preservation,ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>نمونه‌های تشخیص خودکار با A/B/C/D</h2>'+''.join(image(p) for p in illustrations)
    body+='<h2>فرض‌ها و گزارش جامع</h2><pre>'+html.escape(text)+'</pre></html>'
    (folder/'report.html').write_text(body,encoding='utf-8')
    return report,metrics_df,results
