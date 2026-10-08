from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from hippo_lab.data import validate_ticks, make_bars, HISTDATA_TZ
from hippo_lab.engine import Execution, backtest
from hippo_lab.research import features, fit_candidate, signals_for, CANDIDATES, research_run


def ticks(times=None, bid=None, spread=.0001):
    if times is None: times=['2024-01-02 14:00:00','2024-01-02 14:00:01','2024-01-02 14:00:02','2024-01-02 14:00:03']
    if bid is None: bid=[1.2]*len(times)
    return pd.DataFrame({'bid':bid,'ask':np.array(bid)+spread},index=pd.to_datetime(times,utc=True,format='mixed'))


def sig(t, side=1, d=.001, rr=1.):
    return pd.DataFrame({'side':[side],'distance':[d],'rr':[rr]},index=t.index[:1])


def cfg(**kw):
    return replace(Execution(),slippage=0.,latency_ms=0,commission_per_million=0.,tp_trade_through=0.,**kw)

@pytest.mark.parametrize('side',[1,-1])
def test_flat_round_trip_cost_and_reconciliation(side):
    t=ticks(); c=replace(cfg(),slippage=.00002,commission_per_million=35)
    r=backtest(t,sig(t,side),c); trade=r['trades'].iloc[0]
    expected=-trade.units*(.0001+2*.00002+2*35/1e6)
    assert trade.pnl==pytest.approx(expected)
    assert r['equity'].iloc[-1]-c.initial_cash==pytest.approx(r['trades'].pnl.sum())
    assert trade.entry_time>t.index[0]


def test_latency_and_duplicate_timestamps():
    t=ticks(['2024-01-02 14:00:00','2024-01-02 14:00:00','2024-01-02 14:00:00.200','2024-01-02 14:00:00.300','2024-01-02 14:00:01'])
    r=backtest(t,sig(t),replace(cfg(),latency_ms=250))
    assert r['trades'].iloc[0].entry_tick==3


def test_gap_stop_adverse_quote():
    t=ticks(bid=[1.2,1.2,1.198,1.198]); r=backtest(t,sig(t),cfg())
    tr=r['trades'].iloc[0]; assert tr.exit==1.198; assert tr.exit<tr.stop; assert tr.reason==1


def test_short_stop_uses_ask():
    t=ticks(bid=[1.2,1.2,1.20095,1.2]); r=backtest(t,sig(t,-1),cfg())
    assert r['trades'].iloc[0].exit==pytest.approx(1.20105)


def test_tp_cap_and_trade_through():
    t=ticks(bid=[1.2,1.2,1.2011,1.202]); r=backtest(t,sig(t),cfg())
    assert r['trades'].iloc[0].exit==pytest.approx(1.2011)
    r=backtest(t,sig(t),replace(cfg(),tp_trade_through=.0001))
    assert r['trades'].iloc[0].exit_tick==3


def test_stale_entry_skipped():
    t=ticks(['2024-01-02 14:00:00','2024-01-02 14:05:00','2024-01-02 14:05:01'])
    r=backtest(t,sig(t),cfg()); assert len(r['trades'])==0; assert r['stale_or_outside_signals']==1


def test_forced_exit_not_skipped_when_stale():
    t=ticks(['2024-01-02 14:00:00','2024-01-02 14:00:01','2024-01-02 21:50:00','2024-01-02 21:50:01'])
    r=backtest(t,sig(t),cfg(max_hold_hours=24)); assert r['trades'].iloc[0].exit_tick==2
    assert r['trades'].iloc[0].reason==4


def test_marked_dd_breaker_halts_and_gap_overshoot():
    t=ticks(bid=[1.2,1.2,1.15,1.2]); c=cfg(risk_fraction=.1,max_leverage=5,drawdown_breaker=.01)
    r=backtest(t,sig(t,d=.01),c); assert r['halted']; assert r['max_drawdown']>.01
    assert r['trades'].iloc[0].reason==5


def test_bar_boundary_and_partial_last_bar():
    t=ticks(['2024-01-02 14:00:00','2024-01-02 14:14:59','2024-01-02 14:15:00','2024-01-02 14:30:00'],[1.2,1.21,1.4,1.5])
    b=make_bars(t); assert b.loc['2024-01-02 14:15:00+00:00','close']==pytest.approx(1.21005)
    assert b.index[-1]==pd.Timestamp('2024-01-02 14:30',tz='UTC')


def test_quote_validation_and_price_only_rejection():
    t=ticks(); t.iloc[1,1]=1.1
    v,a=validate_ticks(t); assert a['invalid_rows']==1; assert len(v)==3
    with pytest.raises(ValueError):validate_ticks(t.rename(columns={'ask':'price'}))
    with pytest.raises(ValueError):validate_ticks(t.tz_localize(None))
    with pytest.raises(ValueError):backtest(t,sig(t),cfg())


def test_fixed_est_vs_dst():
    naive=ticks().tz_localize(None)
    a,_=validate_ticks(naive,HISTDATA_TZ); assert a.index[0].hour==19
    summer=ticks(['2024-07-02 14:00:00','2024-07-02 14:00:01']).tz_localize(None)
    a,_=validate_ticks(summer,HISTDATA_TZ); b,_=validate_ticks(summer,'America/New_York')
    assert a.index[0].hour==19 and b.index[0].hour==18


def test_currency_restriction():
    with pytest.raises(ValueError):backtest(ticks(),sig(ticks()),replace(cfg(),pair='USDJPY'))


def test_signals_and_refit_truncation_invariance():
    rng=np.random.default_rng(1); idx=pd.date_range('2022-01-01',periods=1800,freq='15min',tz='UTC')
    c=1.2+np.cumsum(rng.normal(0,.0004,len(idx)))
    b=pd.DataFrame({'open':c,'high':c+.0002,'low':c-.0002,'close':c,'spread':.0001},index=idx)
    cut=idx[1300]; altered=b.copy(); altered.loc[cut:,'close']*=5
    np.testing.assert_allclose(features(b).loc[:idx[1299]],features(altered).loc[:idx[1299]],equal_nan=True)
    m1=fit_candidate(CANDIDATES[4],b,idx[0],cut); m2=fit_candidate(CANDIDATES[4],altered,idx[0],cut)
    assert not isinstance(m1,str)
    np.testing.assert_allclose(m1[-1].coef_,m2[-1].coef_)
    np.testing.assert_allclose(m1[0].mean_,m2[0].mean_)
    s1=signals_for(CANDIDATES[0],b,idx[100],cut); s2=signals_for(CANDIDATES[0],altered,idx[100],cut)
    pd.testing.assert_frame_equal(s1,s2)


def test_full_rejects_short_history():
    t=ticks()
    with pytest.raises(ValueError,match='24 complete'):research_run(t,make_bars(t),mode='full')
