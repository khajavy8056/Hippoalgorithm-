import numpy as np
import pandas as pd
from dataclasses import replace
from hippo_lab.structural import StructureRules,detect_structure,directional_pivots


def path():
    rng=np.random.default_rng(44)
    ix=pd.date_range('2020-01-01',periods=5000,freq='min',tz='UTC')
    return pd.Series(100+np.cumsum(rng.normal(0,.1,len(ix))),index=ix)


def test_prefix_invariance_pivots_and_events():
    p=path();r=StructureRules()
    allp=directional_pivots(p,r);past=directional_pivots(p.iloc[:3000],r)
    pd.testing.assert_frame_equal(allp.loc[allp.conf_idx<3000].reset_index(drop=True),past)
    e,_=detect_structure(p,r);ep,_=detect_structure(p.iloc[:3000],r)
    pd.testing.assert_frame_equal(e.loc[e.index<=p.index[2999]],ep)
    assert len(e)>0
    assert (e.index>=e.d_time).all()
    assert (e.c_confirmation_time<=e.d_time).all()


def test_affine_price_scale_invariance():
    p=path();e,_=detect_structure(p);scaled,_=detect_structure(p*100+3000)
    assert e.index.equals(scaled.index)
    np.testing.assert_allclose(e.retrace,scaled.retrace,atol=1e-10)
    np.testing.assert_allclose(e.cd_ab_ratio,scaled.cd_ab_ratio,atol=1e-10)
    np.testing.assert_allclose(scaled.stop_price,e.stop_price*100+3000)


def test_direction_symmetry():
    p=path();e,_=detect_structure(p);rev,_=detect_structure(200-p)
    assert e.index.equals(rev.index)
    np.testing.assert_array_equal(e.side.to_numpy(),-rev.side.to_numpy())
    np.testing.assert_allclose(e.distance,rev.distance)


def test_gap_segments_do_not_cross():
    p=path();ix=p.index.to_list();ix[2500:]=[t+pd.Timedelta(days=2) for t in ix[2500:]];p.index=pd.DatetimeIndex(ix)
    e,l=detect_structure(p)
    assert l['pivots'].segment.nunique()==2
    assert not ((e.a_time<p.index[2500])&(e.index>=p.index[2500])).any()


def test_body_wick_independence():
    p=path();b=pd.DataFrame({'close':p,'open':p,'high':p+1,'low':p-1})
    e,_=detect_structure(b.close)
    b.open=p*1.1;b.high=p*2;b.low=p/2
    other,_=detect_structure(b.close)
    pd.testing.assert_frame_equal(e,other)
