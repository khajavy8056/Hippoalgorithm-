"""Predeclared hypotheses, time-purged ML and nested chronological selection."""
from dataclasses import replace
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .engine import Execution, backtest

CANDIDATES = [
    {'name':'breakout_32', 'family':'breakout', 'lookback':32, 'stop_atr':2., 'rr':1.5},
    {'name':'breakout_64', 'family':'breakout', 'lookback':64, 'stop_atr':2.5, 'rr':2.},
    {'name':'reversion_32', 'family':'reversion', 'lookback':32, 'stop_atr':2., 'rr':1.},
    {'name':'reversion_64', 'family':'reversion', 'lookback':64, 'stop_atr':2.5, 'rr':1.},
    {'name':'logistic_060', 'family':'ml', 'threshold':.60, 'stop_atr':2., 'rr':1.5},
    {'name':'logistic_065', 'family':'ml', 'threshold':.65, 'stop_atr':2., 'rr':1.5},
]
FEATURES = ['ret1','ret4','ret16','z32','trend','vol','cost']
LABEL_HORIZON = 8  # 2 hours on M15
WARMUP_BARS = 96
PROTOCOL = {'candidate_trials':len(CANDIDATES), 'bar_rule':'15min', 'label_horizon':LABEL_HORIZON,
    'minimum_validation_trades':30, 'minimum_ml_independent_labels':100,
    'minimum_months_for_target':24, 'embargo_hours':24, 'minimum_forward_folds':4, 'holdout_months':6, 'selection':'positive net return, DD < 8%, maximize net return / max(DD, 0.5%)',
    'no_passing_candidate':'CASH', 'seed':8056}


def protocol_hash(cfg):
    from dataclasses import asdict
    return hashlib.sha256(json.dumps({'protocol':PROTOCOL, 'candidates':CANDIDATES,
                                      'execution':asdict(cfg)},sort_keys=True).encode()).hexdigest()


def features(b):
    c = b.close; out = pd.DataFrame(index=b.index)
    tr = pd.concat([b.high-b.low,(b.high-c.shift()).abs(),(b.low-c.shift()).abs()],axis=1).max(axis=1)
    out['atr'] = tr.rolling(32,min_periods=32).mean()
    for k in [1,4,16]: out[f'ret{k}'] = c.pct_change(k, fill_method=None)
    out['z32'] = (c-c.rolling(32).mean())/c.rolling(32).std()
    out['trend'] = (c.rolling(16).mean()-c.rolling(64).mean())/out.atr
    out['vol'] = out.ret1.rolling(32).std()
    out['cost'] = b.spread/out.atr
    return out.replace([np.inf,-np.inf],np.nan)


def fit_candidate(candidate, bars, train_start, train_end):
    if candidate['family'] != 'ml': return None
    x = features(bars)
    future = bars.close.shift(-LABEL_HORIZON)/bars.close-1
    label_end = pd.Series(bars.index,index=bars.index).shift(-LABEL_HORIZON)
    # Earliest feature windows entirely inside train; labels end strictly before boundary.
    cutoff=train_end-pd.Timedelta(hours=PROTOCOL['embargo_hours'])
    train = (bars.index >= train_start) & (bars.index < cutoff)
    eligible = train & (label_end < cutoff) & x[FEATURES].notna().all(axis=1)
    eligible[:np.searchsorted(bars.index, train_start)+WARMUP_BARS] = False
    loc = np.flatnonzero(eligible)
    # Non-overlapping labels for training; effective sample count logged/guarded.
    loc = loc[::LABEL_HORIZON]
    if len(loc) < PROTOCOL['minimum_ml_independent_labels'] or (future.iloc[loc] > 0).nunique() < 2:
        return 'INSUFFICIENT'
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, random_state=8056))
    model.fit(x[FEATURES].iloc[loc], (future.iloc[loc]>0).astype(int))
    return model


def signals_for(candidate, bars, start, end, model=None, embargo=True):
    x = features(bars); side = pd.Series(0,index=bars.index,dtype=int)
    if candidate['family'] == 'breakout':
        n = candidate['lookback']
        side.loc[bars.close > bars.high.rolling(n).max().shift(1)] = 1
        side.loc[bars.close < bars.low.rolling(n).min().shift(1)] = -1
    elif candidate['family'] == 'reversion':
        n = candidate['lookback']; z = (bars.close-bars.close.rolling(n).mean())/bars.close.rolling(n).std()
        quiet = x.trend.abs() < .75
        side.loc[(z < -2) & quiet] = 1; side.loc[(z > 2) & quiet] = -1
    elif model is not None and not isinstance(model,str):
        good = x[FEATURES].notna().all(axis=1)
        prob = pd.Series(np.nan,index=bars.index)
        prob.loc[good] = model.predict_proba(x.loc[good,FEATURES])[:,1]
        threshold = candidate['threshold']
        side.loc[prob > threshold] = 1; side.loc[prob < 1-threshold] = -1
    # One fresh setup; no repeated re-entry into the same persistent signal.
    side = side.where(side.ne(side.shift()),0)
    s = pd.DataFrame({'side':side,'distance':x.atr*candidate['stop_atr'],'rr':candidate['rr']})
    allowed = (s.index >= start) & (s.index < end) & (s.side != 0) & (s.distance > 0)
    # Historical bars warm up causal features; no artificial removal of first test days.
    allowed[:WARMUP_BARS] = False
    return s.loc[allowed].dropna()


def empty_signals():
    return pd.DataFrame(columns=['side','distance','rr'], index=pd.DatetimeIndex([],tz='UTC'))


def select(bars,ticks,train_start,val_start,val_end,cfg):
    rows=[]; best=None; best_score=-np.inf
    validation = ticks.loc[(ticks.index >= val_start) & (ticks.index < val_end)]
    if len(validation)<2: raise ValueError('Empty validation period')
    for c in CANDIDATES:
        model=fit_candidate(c,bars,train_start,val_start)
        s=signals_for(c,bars,val_start,val_end,model)
        r=backtest(validation,s,cfg)
        ret=r['equity'].iloc[-1]/cfg.initial_cash-1; dd=r['max_drawdown']; count=len(r['trades'])
        eligible=count>=PROTOCOL['minimum_validation_trades'] and ret>0 and dd<.08
        score=ret/max(dd,.005) if eligible else -np.inf
        rows.append({'candidate':c['name'],'return':ret,'drawdown':dd,'trades':count,
                     'eligible':eligible,'ml_fit': 'insufficient' if isinstance(model,str) else ('trained' if model is not None else 'n/a')})
        if score>best_score: best_score=score; best=c
    return best,pd.DataFrame(rows)


def research_run(ticks,bars,cfg=Execution(),mode='sample'):
    """Sample = smoke test. Full = 12m train/3m validation/3m OOS + final 6m."""
    start=ticks.index[0]; end=ticks.index[-1]; selections=[]; combined=[]
    if mode=='sample':
        # Fixed session boundaries for the pinned sample, not tuned to returns.
        train_start=pd.Timestamp('2016-04-04',tz='America/New_York').tz_convert('UTC')
        val_start=pd.Timestamp('2016-04-06',tz='America/New_York').tz_convert('UTC')
        hold_start=pd.Timestamp('2016-04-07',tz='America/New_York').tz_convert('UTC')
        chosen,table=select(bars,ticks,train_start,val_start,hold_start,cfg)
        selections.append(table.assign(fold='smoke_validation'))
        oos_start=hold_start
    else:
        # First complete UTC month, exclude partial starting month.
        month_start=start.normalize().replace(day=1)
        if start>month_start: month_start += pd.DateOffset(months=1)
        month_end=end.normalize().replace(day=1)
        months=(month_end.year-month_start.year)*12+month_end.month-month_start.month
        if months<24: raise ValueError(f'Need >=24 complete calendar months, have {months}. No target claim possible.')
        # Refuse obvious missing weekday data, rather than silently treating gaps as flat markets.
        dates=pd.date_range(month_start,month_end-pd.Timedelta(days=1),freq='B',tz='UTC')
        observed=ticks.index.normalize().unique()
        missing=dates.difference(observed)
        if len(missing)/max(len(dates),1) > .03:
            raise ValueError(f'Missing {len(missing)} weekdays; inspect data coverage before research')
        ticks=ticks.loc[ticks.index<month_end]
        end=month_end
        hold_start=month_end-pd.DateOffset(months=6)
        oos_start=month_start+pd.DateOffset(months=15)
        cursor=oos_start
        while cursor<hold_start:
            test_end=min(cursor+pd.DateOffset(months=3),hold_start)
            val_start=cursor-pd.DateOffset(months=3); train_start=val_start-pd.DateOffset(months=12)
            c,table=select(bars,ticks,train_start,val_start,cursor,cfg)
            selections.append(table.assign(fold=str(cursor),chosen=c['name'] if c else 'CASH'))
            if c:
                model=fit_candidate(c,bars,train_start,cursor)
                combined.append(signals_for(c,bars,cursor,test_end,model))
            cursor=test_end
        val_start=hold_start-pd.DateOffset(months=3); train_start=val_start-pd.DateOffset(months=12)
        chosen,table=select(bars,ticks,train_start,val_start,hold_start,cfg)
        selections.append(table.assign(fold='final_selection',chosen=chosen['name'] if chosen else 'CASH'))
    # Freeze candidate before accessing final test outcomes. No holdout-driven retries.
    model=fit_candidate(chosen,bars,train_start,hold_start) if chosen else None
    final_signals=signals_for(chosen,bars,hold_start,end,model) if chosen else empty_signals()
    frozen={'candidate':chosen or 'CASH','protocol_hash':protocol_hash(cfg), 'holdout_start':str(hold_start)}
    from pathlib import Path
    model_state=None
    if model is not None and not isinstance(model,str):
        model_state={'coef':model[-1].coef_.tolist(),'intercept':model[-1].intercept_.tolist(),
                     'scaler_mean':model[0].mean_.tolist(),'scaler_scale':model[0].scale_.tolist()}
    frozen['model_state']=model_state
    frozen['selection_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/frozen_selection.json').write_text(json.dumps(frozen,indent=2))
    # Data may already be loaded, but NO holdout outcome feeds selection, refit or this hash.
    final_ticks=ticks.loc[ticks.index>=hold_start]
    final=backtest(final_ticks,final_signals,cfg)
    pre=None
    if mode!='sample':
        pre_ticks=ticks.loc[(ticks.index>=oos_start)&(ticks.index<hold_start)]
        pre_s=pd.concat(combined).sort_index() if combined else empty_signals()
        pre=backtest(pre_ticks,pre_s,cfg)
    stresses={}
    for name,stress_cfg in {
        'base':cfg,
        'spread_1.5_slip_2x':replace(cfg,spread_multiplier=1.5,slippage=cfg.slippage*2),
        'spread_2_slip_3x_latency_1s':replace(cfg,spread_multiplier=2.,slippage=cfg.slippage*3,latency_ms=1000),
    }.items():
        stresses[name]=backtest(final_ticks,final_signals,stress_cfg)
    return {'mode':mode,'selection':pd.concat(selections,ignore_index=True), 'frozen':frozen,
            'final':final,'walk_forward':pre,'stress':stresses,'final_signals':final_signals}
