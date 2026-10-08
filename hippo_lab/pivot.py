"""Causal operational interpretation of the two-page Persian pivot settlement PDF.
Thresholds not specified numerically in the PDF are research assumptions, not recovered rules.
"""
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PivotRules:
    boundary: str = 'wick'  # wick or close; A touch ALWAYS tested on wick
    rr: float = 2.5
    atr_period: int = 14
    ab_max_bars: int = 12
    ab_efficiency: float = .55
    ab_min_median_range_atr: float = .60
    max_noise: int = 1
    retrace_min: float = .20
    retrace_max: float = .50
    bc_min_bars: int = 3
    max_bc_bars: int = 120  # engineering expiry, NOT specified by PDF
    max_armed_bars: int = 120
    signal_body_ratio: float = .60
    signal_range_atr: float = .80
    signal_wick_ratio: float = .25
    b_position_low: float = .20
    b_position_high: float = .80
    trigger_body_atr: float = .50
    stop_buffer_atr: float = .10
    allow_fl: bool = True
    require_time_divergence: bool = False  # optional according to PDF

FEATURE_COLUMNS = ['ab_atr','ab_efficiency','ab_noise','ab_bars','bc_bars','base_bars_after_C','retrace',
                   'cd_ab_ratio','time_divergence','signal_body_ratio','signal_wick_ratio',
                   'signal_range_atr','trigger_body_atr','risk_atr','fl_bars']
EVENT_COLUMNS = ['side','distance','rr','stop_price','setup_id','a_time','b_time','c_time',
                 'cd_start_time','signal_time','a','b','c','d','signal_level',*FEATURE_COLUMNS]


def atr_prior(bars, period=14):
    # Wilder ATR; first period observations warm up; excludes the current candle.
    prev=bars.close.shift()
    tr=pd.concat([bars.high-bars.low,(bars.high-prev).abs(),(bars.low-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean().shift(1)


def pivot_bars(ticks, rule='1min'):
    # Same displayed price convention as quote charts: bid OHLC, not synthetic spread-adjusted mid.
    b=ticks.bid.resample(rule,label='right',closed='left').ohlc().dropna()
    b['count']=ticks.bid.resample(rule,label='right',closed='left').count()
    return b.loc[b.index<=ticks.index[-1]]


def validate_bars(bars):
    if not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError('Closed bars need unique increasing timezone-aware timestamps')
    x=bars[['open','high','low','close']].astype(float)
    if not np.isfinite(x.to_numpy()).all() or (x.high<x[['open','close','low']].max(axis=1)).any() or (x.low>x[['open','close','high']].min(axis=1)).any():
        raise ValueError('Invalid OHLC')
    return x


def _find_ab(o,h,l,c,atr,end,rule):
    """Called only when first pullback bar closes. Longest eligible completed impulse wins."""
    if not np.isfinite(atr[end]) or atr[end]<=0:return None
    for n in range(min(rule.ab_max_bars,end+1),1,-1):
        start=end-n+1; av=atr[end]
        # A and B are candle extremes, with no right-side future fractal confirmation.
        a=l[start]; b=h[end]; amp=b-a
        if amp<=0 or a>np.min(l[start:end+1]) or b<np.max(h[start:end+1]):continue
        noise=int(np.sum(c[start:end+1]<=o[start:end+1]))
        if noise>rule.max_noise:continue
        ranges=h[start:end+1]-l[start:end+1]
        if n==2:
            if not np.all(ranges>av) or noise:continue
        elif n>3 and not 2<=amp/av<=5:continue
        efficiency=(c[end]-o[start])/max(np.sum(ranges),1e-12)
        if efficiency<rule.ab_efficiency or np.median(ranges)/av<rule.ab_min_median_range_atr:continue
        return dict(ai=start,bi=end,a=a,b=b,amp=amp,atr=av,ab_bars=n,
                    ab_noise=noise,ab_efficiency=efficiency)
    return None


def detect_pivots(bars,rules=PivotRules()):
    if len(bars)>1 and ((bars.index.to_series().diff().dropna()<pd.Timedelta(minutes=1)).any()):
        raise ValueError('Pivot implementation requires closed M1 bars, not sub-minute bars')
    if rules.boundary not in ['wick','close']:raise ValueError('boundary = wick or close')
    if rules.rr<=0 or rules.bc_min_bars<3:raise ValueError('Invalid pivot rules')
    x=validate_bars(bars); ats=atr_prior(x,rules.atr_period).to_numpy()
    index=x.index; values=x.to_numpy(); emitted=[]; trace=[]; counts=Counter()
    # M1 contract is explicit; cadence must not be inferred from future timestamps.
    step=pd.Timedelta(minutes=1)
    for direction in [1,-1]:
        # Normalize downward AB by price inversion so every test is expressed as upward AB.
        if direction==1:o,h,l,c=values.T
        else:
            o=-values[:,0]; h=-values[:,2]; l=-values[:,1]; c=-values[:,3]
        state=None; consumed_b=-1
        for i in range(1,len(x)):
            if step is not pd.NaT and index[i]-index[i-1]>3*step:
                if state:counts['reset_data_gap']+=1
                state=None
            if state is None:
                # First pullback must not extend B. Seed at prior completed impulse.
                if c[i]>=c[i-1] or h[i]>h[i-1] or i-1<=consumed_b:continue
                ab=_find_ab(o,h,l,c,ats,i-1,rules)
                if ab is None:continue
                state={**ab,'phase':'BC','ci':i,'c':l[i], 'bc_bars':1,'base_bars':1,'bc_start':i,
                       'highest':h[i-1],'fl_start':None}
                counts['ab_candidates']+=1
                trace.append({'time':str(index[i]),'direction':direction,'stage':'AB_FROZEN','b_time':str(index[i-1])})
                # Seed bar may itself already invalidate BC; process below.
            s=state
            def kill(reason):
                counts[reason]+=1
                trace.append({'time':str(index[i]),'direction':direction,'stage':reason,'b_time':str(index[s['bi']])})
            if l[i]<=s['a']:
                kill('invalid_A_wick_touch');state=None;continue
            if s['phase']=='BC':
                # Keep C provisional throughout the base. Freeze it at the first B wick break.
                # BC duration = B exclusive through FINAL C inclusive, not the entire base.
                breaking=h[i]>s['b']
                if breaking:
                    depth=(s['b']-s['c'])/s['amp']
                    s['bc_bars']=s['ci']-s['bi']
                    if s['bc_bars']<rules.bc_min_bars or depth<rules.retrace_min or (rules.require_time_divergence and s['bc_bars']<=s['ab_bars']):
                        kill('B_break_before_BC_qualified');state=None;continue
                    s['phase']='CD';s['cd_start']=i
                    counts['bc_qualified']+=1
                elif i>s['bc_start']:
                    if l[i]<s['c']:s['ci']=i;s['c']=l[i]
                    s['base_bars']+=1
                    s['bc_bars']=s['ci']-s['bi']
                depth=(s['b']-s['c'])/s['amp']
                if (rules.boundary=='wick' and depth>rules.retrace_max) or c[i]<s['b']-rules.retrace_max*s['amp']:
                    kill('invalid_BC_retrace');state=None;continue
                if s['base_bars']>rules.max_bc_bars:
                    kill('BC_expired');state=None;continue
                if s['phase']=='BC':continue
            bound=s['c']+s['amp']
            # Strict AB>CD cannot touch equal-length boundary; close interpretation allows wick.
            if (rules.boundary=='wick' and h[i]>=bound) or c[i]>=bound:
                kill('invalid_AB_not_gt_CD');state=None;continue
            if s['phase']=='CD' and l[i]<=s['c']:
                # C frozen at breakout; a breakout candle making a new C is rejected.
                kill('invalid_CD_new_C');state=None;continue
            previous_high=s['highest'];s['highest']=max(s['highest'],h[i])
            if s['phase']=='ARMED':
                if i-s['signal_end']>rules.max_armed_bars:
                    kill('trigger_expired');state=None;continue
                av=ats[i]
                if o[i]>=s['signal_low'] and c[i]<s['signal_low'] and o[i]-c[i]>=rules.trigger_body_atr*av:
                    stop=s['highest']+rules.stop_buffer_atr*av
                    risk=stop-c[i]
                    if risk>0:
                        ev={'time':index[i],'side':-direction,'distance':risk,'rr':rules.rr,
                            'stop_price':direction*stop,'setup_id':f'{direction}:{index[s["bi"]].isoformat()}',
                            'a_time':index[s['ai']],'b_time':index[s['bi']],'c_time':index[s['ci']],
                            'cd_start_time':index[s['cd_start']],'signal_time':index[s['signal_end']],
                            'a':direction*s['a'],'b':direction*s['b'],'c':direction*s['c'],
                            'd':direction*s['highest'],'signal_level':direction*s['signal_low'],
                            'ab_atr':s['amp']/s['atr'],'ab_efficiency':s['ab_efficiency'],'ab_noise':s['ab_noise'],
                            'ab_bars':s['ab_bars'],'bc_bars':s['bc_bars'],'base_bars_after_C':s['base_bars']-s['bc_bars'],'retrace':(s['b']-s['c'])/s['amp'],
                            'cd_ab_ratio':(s['highest']-s['c'])/s['amp'],
                            'time_divergence':float(s['bc_bars']>s['ab_bars']),
                            'signal_body_ratio':s['body_ratio'],'signal_wick_ratio':s['wick_ratio'],
                            'signal_range_atr':s['sig_range_atr'],'trigger_body_atr':(o[i]-c[i])/av,
                            'risk_atr':risk/av,'fl_bars':s['fl_bars']}
                        emitted.append(ev);counts['triggers']+=1;consumed_b=s['bi']
                    state=None
                continue
            if s['fl_start'] is None:
                if h[i]>s['b'] and h[i]>previous_high:
                    s['fl_start']=i
                else:continue
            k=s['fl_start'];fl_n=i-k+1
            if fl_n>(3 if rules.allow_fl else 1):
                kill('invalid_signal_or_FL');state=None;continue
            high=float(np.max(h[k:i+1])); low=float(np.min(l[k:i+1]));rng=high-low
            body=c[i]-o[k];upper=high-max(o[k],c[i]);pos=(s['b']-low)/max(rng,1e-12)
            body_ratio=body/max(rng,1e-12);wick_ratio=upper/max(rng,1e-12)
            if body>0 and c[i]>s['b'] and body_ratio>=rules.signal_body_ratio and wick_ratio<=rules.signal_wick_ratio and rng>=rules.signal_range_atr*ats[k] and rules.b_position_low<=pos<=rules.b_position_high and (fl_n==1 or high>h[k]):
                s.update(phase='ARMED',signal_end=i,signal_low=low,body_ratio=body_ratio,
                         wick_ratio=wick_ratio,sig_range_atr=rng/ats[k],fl_bars=fl_n)
                counts['signals_armed']+=1
                trace.append({'time':str(index[i]),'direction':direction,'stage':'ARMED','b_time':str(index[s['bi']])})
    events=pd.DataFrame(emitted)
    if len(events):
        events=events.set_index('time').sort_index(kind='stable')
        # A simultaneous opposite trigger is ambiguous; drop BOTH, not choose hindsight winner.
        collision=events.index.duplicated(keep=False);counts['simultaneous_dropped']=int(collision.sum())
        events=events.loc[~collision]
    else:events=pd.DataFrame(columns=EVENT_COLUMNS,index=pd.DatetimeIndex([],tz='UTC',name='time'))
    return events, {'rules':asdict(rules),'counts':dict(counts),'trace':trace}
