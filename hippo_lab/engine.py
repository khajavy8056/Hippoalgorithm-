"""Single-position event-driven quote replay; account currency = pair quote currency."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from numba import njit

@dataclass(frozen=True)
class Execution:
    pair: str = "GBPUSD"
    account_currency: str = "USD"
    initial_cash: float = 10000.0
    risk_fraction: float = .0025
    max_leverage: float = 5.0
    slippage: float = .00002  # absolute quote-price units per fill
    commission_per_million: float = 35.0  # per SIDE, quote currency / million BASE units
    spread_multiplier: float = 1.0
    tp_trade_through: float = .00001  # require executable quote beyond limit
    max_spread: float = .00030
    latency_ms: int = 250
    max_entry_delay_s: int = 60
    max_hold_hours: float = 6.0
    drawdown_breaker: float = .08
    daily_loss_limit: float = .02
    session_start_hour: int = 7
    session_end_hour: int = 16
    flatten_hour: float = 16.75  # NY, before rollover; no overnight swap model

@njit
def _replay(ts, bid0, ask0, local_day, hour, signal_tick, sides, distances, rr,
            cash0, risk, leverage, slip, fee, spread_mult, max_spread,
            hold_ns, breaker, daily_limit, tp_through, session_start, session_end, flatten_hour):
    n = len(ts); eq = np.empty(n); ledger = np.empty((len(sides)+1, 9))
    cash = cash0; peak = cash0; day_cash = cash0; day = local_day[0]
    pos = 0; units = 0.; entry = 0.; stop = 0.; target = 0.; entry_i = 0
    ptr = 0; trades = 0; halted = False; day_halt = False; max_dd = 0.
    for i in range(n):
        spread = (ask0[i]-bid0[i])*spread_mult
        mid = (ask0[i]+bid0[i])/2
        bid = mid-spread/2; ask = mid+spread/2
        liquidation = bid if pos == 1 else ask
        mark = cash + pos*units*(liquidation-entry) - (units*fee if pos else 0.)
        if local_day[i] != day:
            day = local_day[i]; day_cash = mark; day_halt = False
        peak = max(peak, mark); dd = max(0., 1-mark/peak); max_dd = max(max_dd, dd)
        if dd >= breaker: halted = True
        if mark <= day_cash*(1-daily_limit): day_halt = True
        exited = False
        if pos:
            reason = 0
            if (pos == 1 and bid <= stop) or (pos == -1 and ask >= stop): reason = 1
            elif (pos == 1 and bid >= target+tp_through) or (pos == -1 and ask <= target-tp_through): reason = 2
            elif ts[i]-ts[entry_i] >= hold_ns: reason = 3
            elif hour[i] >= flatten_hour or local_day[i] != local_day[entry_i]: reason = 4
            if halted or day_halt: reason = 5
            if i == n-1: reason = 6
            if reason:
                # Stop gaps filled at the observed executable quote, not at requested stop.
                # Take-profit capped at its limit; no favourable price improvement assumed.
                fill = liquidation - pos*slip
                if reason == 2: fill = min(fill, target) if pos == 1 else max(fill, target)
                pnl = pos*units*(fill-entry) - 2*units*fee
                cash += pos*units*(fill-entry) - units*fee
                ledger[trades] = np.array([entry_i,i,pos,units,entry,fill,pnl,reason,stop])
                trades += 1; pos = 0; exited = True; mark = cash
        # Signals arriving while busy are discarded, NOT queued for a later entry.
        while ptr < len(sides) and signal_tick[ptr] <= i:
            if signal_tick[ptr] == i and not pos and not exited and not halted and not day_halt and i < n-1:
                d = distances[ptr]; side = sides[ptr]
                if d > spread+2*slip and spread <= max_spread and session_start <= hour[i] < min(session_end,flatten_hour):
                    price = (ask if side == 1 else bid) + side*slip
                    # Include expected stop exit slippage and both commissions in risk budget.
                    qty = min(cash*risk/(d+slip+2*fee), cash*leverage/price)
                    if qty > 0:
                        pos = side; units = qty; entry = price; entry_i = i
                        stop = price-side*d; target = price+side*d*rr[ptr]
                        cash -= units*fee
                        mark = cash + pos*units*((bid if pos == 1 else ask)-entry)-units*fee
            ptr += 1
        eq[i] = mark
        peak = max(peak, mark); max_dd = max(max_dd, max(0., 1-mark/peak))
    return eq, ledger[:trades], max_dd, halted


def backtest(ticks, signals, cfg=Execution()):
    if len(cfg.pair) != 6 or cfg.account_currency != cfg.pair[-3:]:
        raise ValueError('Account currency must equal quote currency; cross-currency conversion is not implemented')
    if cfg.initial_cash <= 0 or not 0 < cfg.risk_fraction < 1 or cfg.slippage < 0 or cfg.spread_multiplier < 1:
        raise ValueError('Invalid execution configuration')
    if not ticks.index.is_monotonic_increasing or ticks.index.tz is None:
        raise ValueError('Sorted timezone-aware quotes required')
    if (ticks.ask < ticks.bid).any(): raise ValueError('Crossed quotes')
    if len(ticks) < 2: raise ValueError('At least two quotes required')
    ts = ticks.index.as_unit('ns').asi8
    s = signals.sort_index(kind='stable')
    if len(s) and (not s.index.is_unique or not s.side.isin([-1,1]).all() or (s.distance <= 0).any() or (s.rr <= 0).any()):
        raise ValueError('Signals must have unique times, side +/-1 and positive distance/rr')
    ready = s.index.as_unit('ns').asi8 + cfg.latency_ms*1_000_000
    # Strictly later timestamp even at zero configured latency.
    inds = np.maximum(np.searchsorted(ts, ready, side='left'), np.searchsorted(ts, s.index.as_unit('ns').asi8, side='right'))
    valid = inds < len(ts)
    safe = np.minimum(inds, len(ts)-1)
    valid &= ts[safe]-ready <= cfg.max_entry_delay_s*1_000_000_000
    local = ticks.index.tz_convert('America/New_York')
    days = np.asarray(local.year*10000+local.month*100+local.day, dtype=np.int64)
    hours = np.asarray(local.hour+local.minute/60+local.second/3600, dtype=float)
    equity, ledger, max_dd, halted = _replay(ts, ticks.bid.to_numpy(), ticks.ask.to_numpy(), days, hours,
        inds[valid], s.side.to_numpy(dtype=np.int64)[valid], s.distance.to_numpy(dtype=float)[valid], s.rr.to_numpy(dtype=float)[valid],
        cfg.initial_cash, cfg.risk_fraction, cfg.max_leverage, cfg.slippage, cfg.commission_per_million/1e6,
        cfg.spread_multiplier, cfg.max_spread, int(cfg.max_hold_hours*3600e9), cfg.drawdown_breaker,
        cfg.daily_loss_limit, cfg.tp_trade_through, cfg.session_start_hour, cfg.session_end_hour, cfg.flatten_hour)
    trades = pd.DataFrame(ledger, columns=['entry_tick','exit_tick','side','units','entry','exit','pnl','reason','stop'])
    if len(trades):
        gap_seconds=np.r_[0.,np.diff(ts)/1e9]
        gap_prefix=np.cumsum(gap_seconds>60)
        trades['quote_gap_over_60s']=gap_prefix[trades.exit_tick.astype(int)]>gap_prefix[trades.entry_tick.astype(int)]
        trades['planned_loss'] = trades.units*((trades.entry-trades.stop).abs()+cfg.slippage+2*cfg.commission_per_million/1e6)
        trades['realized_R'] = trades.pnl/trades.planned_loss
        trades['entry_time'] = ticks.index[trades.entry_tick.astype(int)]
        trades['exit_time'] = ticks.index[trades.exit_tick.astype(int)]
    return {'equity':pd.Series(equity,index=ticks.index,name='equity'), 'trades':trades,
            'max_drawdown':float(max_dd), 'halted':bool(halted), 'config':asdict(cfg),
            'submitted_signals':len(s), 'stale_or_outside_signals':int((~valid).sum())}
