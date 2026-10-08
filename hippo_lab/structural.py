"""Price-path pivot settlement. No candle body/wick/color/count conditions.
Close-only observations remain time-sampled; this is NOT a tick-invariant strategy.
Every pivot records occurrence and confirmation separately; execution only after confirmation.
"""
from dataclasses import dataclass,asdict
from collections import Counter
import numpy as np
import pandas as pd
from numba import njit

@dataclass(frozen=True)
class StructureRules:
    dc_multiple: float = 3.0
    volatility_span: int = 60
    warmup: int = 60
    gap_reset_seconds: float = 180.
    ab_min_thresholds: float = 2.0
    efficiency_min: float = .45
    retrace_min: float = .20
    retrace_max: float = .50
    cd_ab_max: float = 1.0
    require_return_inside_B: bool = True
    stop_buffer_thresholds: float = .10
    rr: float = 2.5

FEATURES=['ab_thresholds','efficiency','retrace','cd_ab_ratio','confirmation_lag_seconds',
          'late_C_confirmation','risk_thresholds','direction']

@njit
def _pivots(price,ts,k,span,warmup,gap):
    # output: occurrence idx, confirmation idx, price, type(+1 high/-1 low),
    # threshold frozen at START of leg, path length at extreme, segment.
    out=np.empty((len(price),7));count=0;direction=0;extreme=price[0];ei=0
    low=price[0];li=0;high=price[0];hi=0
    vol=0.;samples=0;theta=0.;path=0.;extreme_path=0.;segment=0
    alpha=2./(span+1)
    for i in range(1,len(price)):
        if (ts[i]-ts[i-1])/1e9>gap:
            segment+=1;direction=0;extreme=price[i];ei=i;low=price[i];li=i;high=price[i];hi=i
            vol=0.;samples=0;theta=0.;path=0.;extreme_path=0.;continue
        delta=abs(price[i]-price[i-1]);path+=delta
        # Prior EWMA only. Once a leg starts, threshold stays frozen until confirmation.
        if samples>=warmup and vol>0:
            if theta==0:
                theta=k*vol;low=price[i-1];li=i-1;high=low;hi=li
            if direction==0:
                if price[i]<low:low=price[i];li=i
                if price[i]>high:high=price[i];hi=i
                if price[i]-low>=theta:
                    out[count]=np.array([li,i,low,-1.,theta,path,segment]);count+=1
                    direction=1;extreme=price[i];ei=i;path=0.;extreme_path=0.;theta=k*vol
                elif high-price[i]>=theta:
                    out[count]=np.array([hi,i,high,1.,theta,path,segment]);count+=1
                    direction=-1;extreme=price[i];ei=i;path=0.;extreme_path=0.;theta=k*vol
            elif direction==1:
                if price[i]>extreme:extreme=price[i];ei=i;extreme_path=path
                elif extreme-price[i]>=theta:
                    out[count]=np.array([ei,i,extreme,1.,theta,extreme_path,segment]);count+=1
                    direction=-1;extreme=price[i];ei=i;path=0.;extreme_path=0.;theta=k*vol
            else:
                if price[i]<extreme:extreme=price[i];ei=i;extreme_path=path
                elif price[i]-extreme>=theta:
                    out[count]=np.array([ei,i,extreme,-1.,theta,extreme_path,segment]);count+=1
                    direction=1;extreme=price[i];ei=i;path=0.;extreme_path=0.;theta=k*vol
        vol=delta if samples==0 else alpha*delta+(1-alpha)*vol;samples+=1
    return out[:count]


def directional_pivots(prices,rules=StructureRules()):
    if not isinstance(prices.index,pd.DatetimeIndex) or prices.index.tz is None or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Unique sorted timezone-aware price observations required')
    v=prices.to_numpy(dtype=float)
    if len(v)<2 or not np.isfinite(v).all():raise ValueError('Finite price path of length >=2 required')
    if rules.dc_multiple<=0 or rules.volatility_span<2 or rules.warmup<2:raise ValueError('Invalid structure rules')
    arr=_pivots(v,prices.index.as_unit('ns').asi8,rules.dc_multiple,rules.volatility_span,rules.warmup,rules.gap_reset_seconds)
    f=pd.DataFrame(arr,columns=['occ_idx','conf_idx','price','kind','threshold','leg_path','segment'])
    f['occurrence_time']=prices.index[f.occ_idx.astype(int)]
    f['confirmation_time']=prices.index[f.conf_idx.astype(int)]
    return f


def detect_structure(prices,rules=StructureRules()):
    piv=directional_pivots(prices,rules);counts=Counter(confirmed_pivots=len(piv));rows=[]
    cum=np.r_[0.,np.cumsum(np.abs(np.diff(prices.to_numpy(dtype=float))))]
    for j in range(3,len(piv)):
        a,b,c,d=[piv.iloc[q] for q in range(j-3,j+1)]
        if a.segment!=d.segment:continue
        direction=1 if a.kind==-1 else -1
        A,B,C,D=direction*np.array([a.price,b.price,c.price,d.price])
        ab=B-A;bc=B-C;cd=D-C
        counts['ABCD_candidates']+=1
        path=cum[int(b.occ_idx)]-cum[int(a.occ_idx)]
        eff=ab/max(path,1e-15);ab_t=ab/b.threshold
        if ab_t<rules.ab_min_thresholds or eff<rules.efficiency_min:
            counts['reject_impulse']+=1;continue
        depth=bc/max(ab,1e-15)
        if not rules.retrace_min<=depth<=rules.retrace_max or C<=A:
            counts['reject_retrace']+=1;continue
        counts['qualified_ABC']+=1
        # C confirmation, not occurrence, must precede D occurrence; late-break flagged separately.
        if int(c.conf_idx)>int(d.occ_idx):counts['reject_late_confirmation']+=1;continue
        if D<=B:counts['reject_no_B_break']+=1;continue
        counts['B_broken']+=1
        if cd>=rules.cd_ab_max*ab:
            counts['reject_CD_size']+=1;continue
        counts['divergent_CD']+=1
        trigger=float(prices.iloc[int(d.conf_idx)])*direction
        if trigger<=C:counts['reject_C_breached_at_trigger']+=1;continue
        if rules.require_return_inside_B and trigger>=B:
            counts['reject_not_back_inside_B']+=1;continue
        stop=D+rules.stop_buffer_thresholds*d.threshold;risk=stop-trigger
        time=prices.index[int(d.conf_idx)]
        rows.append({'time':time,'side':-direction,'distance':risk,'rr':rules.rr,
                     'stop_price':direction*stop,'setup_id':f'DC:{direction}:{b.confirmation_time}',
                     'a':direction*A,'b':direction*B,'c':direction*C,'d':direction*D,
                     'a_time':a.occurrence_time,'b_time':b.occurrence_time,'c_time':c.occurrence_time,
                     'd_time':d.occurrence_time,'c_confirmation_time':c.confirmation_time,
                     'signal_time':d.confirmation_time,'ab_thresholds':ab_t,'efficiency':eff,
                     'retrace':depth,'cd_ab_ratio':cd/ab,'threshold':d.threshold,
                     'confirmation_lag_seconds':(d.confirmation_time-d.occurrence_time).total_seconds(),
                     'late_C_confirmation':float(direction*prices.iloc[int(c.conf_idx)]>B),
                     'risk_thresholds':risk/d.threshold,'direction':direction})
    cols=['side','distance','rr','stop_price','setup_id','a','b','c','d','a_time','b_time','c_time',
          'd_time','c_confirmation_time','signal_time',*FEATURES,'threshold']
    events=pd.DataFrame(rows).set_index('time') if rows else pd.DataFrame(columns=cols,index=pd.DatetimeIndex([],tz='UTC',name='time'))
    counts['triggers']=len(events)
    return events,{'rules':asdict(rules),'counts':dict(counts),'pivots':piv}
