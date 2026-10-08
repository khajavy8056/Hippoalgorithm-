"""Compare exact OHLC exports to user annotations; proximity is not proof of fidelity."""
from pathlib import Path
import pandas as pd
from .pivot import validate_bars,detect_pivots,PivotRules


def read_chart_bars(path, source_timezone, timestamp_is_open=True):
    f=pd.read_csv(path);f.columns=[str(c).lower() for c in f.columns]
    if 'time' in f and 'timestamp' not in f:f=f.rename(columns={'time':'timestamp'})
    if 'timestamp' not in f:raise ValueError('CSV requires timestamp/time and open/high/low/close')
    raw=f.pop('timestamp')
    if pd.api.types.is_numeric_dtype(raw):idx=pd.DatetimeIndex(pd.to_datetime(raw,unit='s',utc=True))
    else:idx=pd.DatetimeIndex(pd.to_datetime(raw,format='mixed'))
    if idx.tz is None:idx=idx.tz_localize(source_timezone,ambiguous='raise',nonexistent='raise')
    idx=idx.tz_convert('UTC')
    if timestamp_is_open:idx=idx+pd.Timedelta(minutes=1)
    f.index=idx;return validate_bars(f)


def compare_images(bars,annotations,chart_timezone,rules=PivotRules(),tolerance_minutes=2):
    events,log=detect_pivots(bars,rules)
    records=[]
    for _,a in annotations.iterrows():
        time=a.get('approx_trigger_chart_local')
        base={'image':a.image,'boundary':rules.boundary,'expected_side':a.side,'status':'UNVERIFIABLE',
              'nearest_trigger':None,'delta_minutes':None,'note':'Approximate screenshot timing; needs manual OHLC review.'}
        if pd.isna(time) or not time:
            base['note']='No readable trigger timestamp in screenshot';records.append(base);continue
        expected=pd.Timestamp(str(a.chart_date)+' '+str(time)).tz_localize(chart_timezone).tz_convert('UTC')
        # Chart mark is approximate entry/open time, detector event is trigger candle close.
        subset=events.loc[(events.index>=expected-pd.Timedelta(minutes=tolerance_minutes)) &
                          (events.index<=expected+pd.Timedelta(minutes=tolerance_minutes))]
        if not (bars.index.min()<=expected<=bars.index.max()):
            base['status']='DATA_DATE_NOT_COVERED'
        elif len(subset)==0:base['status']='NO_NEARBY_TRIGGER'
        else:
            side=-1 if a.side=='sell' else 1;same=subset.loc[subset.side==side]
            if len(same):
                delta=abs((same.index-expected).total_seconds()/60);j=int(delta.argmin())
                base.update(status='NEARBY_SAME_DIRECTION_NOT_EXACT_PROOF',nearest_trigger=str(same.index[j]),delta_minutes=float(delta[j]))
            else:base['status']='ONLY_OPPOSITE_TRIGGER_NEARBY'
        records.append(base)
    return pd.DataFrame(records),events,log
