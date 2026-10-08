from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from hippo_lab.pivot import detect_pivots,PivotRules,FEATURE_COLUMNS
from hippo_lab.pivot_experiment import paired_retention,nonoverlapping,fit_filter
from hippo_lab.engine import Execution,backtest


def fixture():
    # Artificial OHLC ONLY a unit test, never financial performance evidence.
    rows=[(100,100.5,99.5,100)]*8
    rows += [(100,102.2,99.8,102),(102,104.2,101.8,104),(104,106.2,103.8,106)]
    rows += [(106,106.1,105,105.5),(105.5,105.8,104.8,105),(105,105.2,104,104.5),
             (104.5,107,104.3,106.8),(106.8,107,103.5,103.8),(103.8,104,103,103.2)]
    ix=pd.date_range('2024-01-02 14:00',periods=len(rows),freq='1min',tz='UTC')
    return pd.DataFrame(rows,columns=['open','high','low','close'],index=ix)


def rules(**kw):
    return replace(PivotRules(),atr_period=2,ab_max_bars=3,ab_min_median_range_atr=.1,
                   signal_range_atr=.1,trigger_body_atr=.1,**kw)


def test_AB_CD_then_reversal_not_breakout_entry():
    b=fixture();e,log=detect_pivots(b,rules())
    assert len(e)==1,log['counts']
    t=e.iloc[0]; assert t.side==-1; assert e.index[0]==b.index[15]
    assert t.signal_time==b.index[14];assert t.b_time==b.index[10];assert t.c_time==b.index[13]
    assert t.bc_bars==3;assert t.cd_ab_ratio<1
    assert t.stop_price>107


def test_long_short_symmetry():
    b=fixture();mirror=pd.DataFrame({'open':200-b.open,'high':200-b.low,'low':200-b.high,'close':200-b.close},index=b.index)
    e1,_=detect_pivots(b,rules());e2,_=detect_pivots(mirror,rules())
    assert len(e2)==1;assert e2.iloc[0].side==1
    assert e2.iloc[0].stop_price==pytest.approx(200-e1.iloc[0].stop_price)
    assert e2.iloc[0].distance==pytest.approx(e1.iloc[0].distance)


def test_boundary_interpretations_wick_vs_close():
    b=fixture();b.loc[b.index[14],'high']=111 # beyond C+AB=110.4, close stays inside
    # signal quality relaxed to isolate boundary check, not production defaults
    e1,l1=detect_pivots(b,rules(boundary='wick',signal_wick_ratio=1.,signal_body_ratio=.01))
    e2,l2=detect_pivots(b,rules(boundary='close',signal_wick_ratio=1.,signal_body_ratio=.01))
    assert len(e1)==0 and l1['counts']['invalid_AB_not_gt_CD']>=1
    assert len(e2)==1


def test_bc_wick_beyond_half_close_inside():
    b=fixture();b.loc[b.index[13],'low']=102.8 # deeper than half but close above 103
    e1,_=detect_pivots(b,rules(boundary='wick'));e2,_=detect_pivots(b,rules(boundary='close'))
    assert len(e1)==0;assert len(e2)==1


def test_A_wick_touch_always_invalid():
    b=fixture();b.loc[b.index[13],'low']=99.8
    for boundary in ['wick','close']:
        e,log=detect_pivots(b,rules(boundary=boundary));assert len(e)==0;assert log['counts']['invalid_A_wick_touch']>=1


def test_prefix_invariance():
    b=fixture();e,_=detect_pivots(b,rules());prefix,_=detect_pivots(b.iloc[:16],rules())
    pd.testing.assert_frame_equal(e,prefix)
    b.iloc[16]=[103,130,80,100];altered,_=detect_pivots(b,rules())
    pd.testing.assert_frame_equal(e,altered)


def test_absolute_stop_not_shifted_on_actual_entry():
    ix=pd.date_range('2024-01-02 14:00',periods=4,freq='s',tz='UTC')
    t=pd.DataFrame({'bid':[1.2,1.2003,1.201,1.2011],'ask':[1.2001,1.2004,1.2011,1.2012]},index=ix)
    sig=pd.DataFrame({'side':[-1],'distance':[.001],'rr':[2.5],'stop_price':[1.201]},index=ix[:1])
    c=replace(Execution(),slippage=0,commission_per_million=0,latency_ms=0)
    r=backtest(t,sig,c);trade=r['trades'].iloc[0]
    assert trade.stop==1.201;assert trade.entry==1.2003;assert trade.reason==1
    assert r['equity'].iloc[-1]-c.initial_cash==pytest.approx(trade.pnl)


def test_ml_rejects_insufficient_and_reports_removed_winners():
    ix=pd.date_range('2024-01-01',periods=4,freq='1D',tz='UTC')
    labels=pd.DataFrame({'pnl':[10,-5,20,-8],'label_exit':ix+pd.Timedelta(hours=1)},index=ix)
    m=paired_retention(labels,[True,False,False,True])
    assert m['removed_winners']==1 and m['removed_positive_pnl']==20
    assert m['winner_retention']==.5;assert m['avoided_negative_pnl']==5
    model,th,table,info=fit_filter(labels,labels,ix[2],ix[3],False)
    assert model is None and th is None and not info['enabled']


def test_nonoverlap_label_purge():
    ix=pd.date_range('2024-01-01',periods=4,freq='1h',tz='UTC')
    labels=pd.DataFrame({'label_exit':[ix[2],ix[2],ix[3],ix[3]+pd.Timedelta(minutes=1)]},index=ix)
    assert len(nonoverlapping(labels))==2


def test_FL_two_bar_combination_and_no_same_bar_trigger():
    b=fixture(); rows=b.iloc[:14].copy()
    # First hunt too small in body; next extends first high and combines into strong signal.
    r1=pd.DataFrame([[105.7,106.5,105.5,106.1],[106.1,107,106.,106.9],
                     [106.9,107,105.2,105.4]],columns=b.columns,
                    index=pd.date_range(b.index[14],periods=3,freq='min'))
    b=pd.concat([rows,r1]);e,log=detect_pivots(b,rules())
    assert len(e)==1,log['counts'];assert e.iloc[0].fl_bars==2
    assert e.index[0]>e.iloc[0].signal_time
    assert e.iloc[0].signal_level==105.5


def test_gap_resets_setup_not_repaired_by_future():
    b=fixture();ix=b.index.to_list();ix[14:]=[t+pd.Timedelta(hours=1) for t in ix[14:]];b.index=pd.DatetimeIndex(ix)
    e,log=detect_pivots(b,rules());assert len(e)==0;assert log['counts']['reset_data_gap']>=1


def test_filter_fitting_and_thresholds_never_use_test_labels():
    rng=np.random.default_rng(22);ix=pd.date_range('2020-01-01',periods=260,freq='D',tz='UTC')
    labels=pd.DataFrame(rng.normal(size=(260,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS,index=ix)
    labels['pnl']=np.where(labels.ab_atr>0,10.,-10.)
    labels['label_exit']=ix+pd.Timedelta(hours=1)
    changed=labels.copy();changed.loc[ix[220]:,'pnl']*=-50
    m1,t1,tab1,i1=fit_filter(labels,labels,ix[150],ix[220],True)
    m2,t2,tab2,i2=fit_filter(changed,changed,ix[150],ix[220],True)
    assert m1 is not None and i1['enabled']
    assert t1==t2
    np.testing.assert_allclose(m1[-1].coef_,m2[-1].coef_)
    pd.testing.assert_frame_equal(tab1,tab2)


def test_long_quote_gate_rejects_short_and_ohlc():
    from hippo_lab.pivot_long import long_quote_gate
    ix=pd.date_range('2024-01-01',periods=3,freq='min',tz='UTC')
    quotes=pd.DataFrame({'bid':1.2,'ask':1.2001},index=ix)
    with pytest.raises(ValueError,match='365'):long_quote_gate(quotes)
    with pytest.raises(ValueError,match='bid AND ask'):long_quote_gate(fixture())
