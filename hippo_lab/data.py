"""Quote loaders. No price-only fallback and no synthetic quotes."""
from pathlib import Path
from datetime import timezone, timedelta
import hashlib, io, zipfile, time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

HISTDATA_TZ = timezone(timedelta(hours=-5))  # fixed EST, NOT America/New_York
SAMPLE_COMMIT = '1e0879d8214bafffe44d48e61d1ec6410ec4e7ec'


def validate_ticks(frame, naive_tz=None):
    f = frame.rename(columns={c: str(c).lower() for c in frame.columns}).copy()
    if 'timestamp' in f:
        f.index = pd.DatetimeIndex(pd.to_datetime(f.pop('timestamp'), format='mixed'))
    if not isinstance(f.index, pd.DatetimeIndex):
        raise ValueError('DatetimeIndex or timestamp column required')
    if f.index.tz is None:
        if naive_tz is None:
            raise ValueError('Naive timestamps: explicitly provide source timezone')
        f.index = f.index.tz_localize(naive_tz, ambiguous='raise', nonexistent='raise')
    f.index = f.index.tz_convert('UTC').as_unit('ns')
    if not {'bid', 'ask'} <= set(f):
        raise ValueError('Real bid AND ask required. Price-only data are not executable quotes.')
    f = f[['bid', 'ask']].astype(float)
    good = np.isfinite(f).all(axis=1) & (f.bid > 0) & (f.ask >= f.bid)
    audit = {'input_rows': len(f), 'invalid_rows': int((~good).sum()),
             'out_of_order_rows': int((f.index.to_series().diff().dt.total_seconds() < 0).sum()),
             'duplicate_timestamps': int(f.index.duplicated().sum())}
    # Preserve differing quotes with equal timestamps, and their original source order.
    f = f.loc[good].sort_index(kind='stable')
    if len(f) < 2:
        raise ValueError('Insufficient valid quotes')
    gaps = f.index.to_series().diff().dt.total_seconds()
    audit.update(valid_rows=len(f), start=str(f.index[0]), end=str(f.index[-1]),
                 gaps_over_60s=int((gaps > 60).sum()), gaps_over_5min=int((gaps > 300).sum()), max_gap_seconds=float(gaps.max()),
                 median_spread=float((f.ask-f.bid).median()),
                 p99_spread=float((f.ask-f.bid).quantile(.99)))
    return f, audit


def read_ticks(path, naive_tz=HISTDATA_TZ):
    p = Path(path)
    if p.suffix == '.parquet':
        return validate_ticks(pd.read_parquet(p), naive_tz)
    if p.suffix == '.zip':
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if len(names) != 1:
                raise ValueError('Choose a ZIP containing exactly one quote CSV')
            with z.open(names[0]) as stream:
                f = pd.read_csv(stream, header=None, names=['timestamp','bid','ask','volume'])
        f.timestamp = pd.to_datetime(f.timestamp, format='%Y%m%d %H%M%S%f')
        return validate_ticks(f, naive_tz)
    return validate_ticks(pd.read_csv(p), naive_tz)


def download_histdata(pair, year, month, folder='data/histdata', retries=3):
    """Same form-token + POST method as the supplied Colab, fail loudly."""
    pair = pair.replace('/', '').upper()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{pair}_{year}_{month:02d}.zip'
    if dest.exists():
        read_ticks(dest)  # never silently trust a partial archive
        return dest
    page = f'https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{pair.lower()}/{year}/{month}'
    errors = []
    for attempt in range(retries):
        try:
            with requests.Session() as s:
                s.headers.update({'User-Agent':'Mozilla/5.0', 'Referer':page})
                r = s.get(page, timeout=45); r.raise_for_status()
                form = BeautifulSoup(r.content, 'html.parser').find('form', id='file_down')
                if form is None: raise ValueError('HistData token form not found')
                payload = {t.get('name'): t.get('value','') for t in form.find_all('input') if t.get('name')}
                r = s.post('https://www.histdata.com/get.php', data=payload, timeout=120)
                r.raise_for_status()
                if not zipfile.is_zipfile(io.BytesIO(r.content)): raise ValueError('Response is not a ZIP')
                tmp = dest.with_suffix('.partial.zip'); tmp.write_bytes(r.content)
                read_ticks(tmp); tmp.replace(dest)
                return dest
        except (requests.RequestException, ValueError, zipfile.BadZipFile) as e:
            errors.append(str(e)); time.sleep(2**attempt)
    raise RuntimeError(f'Failed {pair} {year}-{month:02d}: {errors}; upload original quote ZIP/parquet instead')


def download_sample(folder='data/cache'):
    """A public GAIN Capital mirror: provenance claimed by repo, not broker-certified."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    p = folder/'gain_sample.csv'
    if not p.exists():
        url = f'https://codeload.github.com/phuocidi/cleaned_tick_data/zip/{SAMPLE_COMMIT}'
        r = requests.get(url, timeout=90); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            n = next(n for n in z.namelist() if n.endswith('2016_04_GBP_USD_Week1.csv'))
            p.write_bytes(z.read(n))
    f = pd.read_csv(p, header=None, names=['pair','timestamp','bid','ask'])
    # GAIN historical time is New York local (DST applies); verify against broker before deployment.
    ticks, audit = validate_ticks(f, 'America/New_York')
    audit.update(source='Public mirror of GAIN Capital GBPUSD Week1 April 2016',
                 commit=SAMPLE_COMMIT, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                 timezone_assumption='America/New_York; not independently broker-certified')
    return ticks, audit


def make_bars(ticks, rule='15min'):
    mid = (ticks.bid + ticks.ask)/2
    bars = mid.resample(rule, label='right', closed='left').ohlc()
    bars['count'] = mid.resample(rule, label='right', closed='left').count()
    bars['spread'] = (ticks.ask-ticks.bid).resample(rule, label='right', closed='left').median()
    bars = bars.dropna()
    # Do not treat the final unfinished candle as closed.
    return bars.loc[bars.index <= ticks.index[-1]]
