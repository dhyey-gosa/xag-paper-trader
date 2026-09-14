"""
Paper Trading Web App: XAG/USDT Combined Breakout+Momentum Strategy
Flask backend with live Binance data + dashboard
"""
import os
import sys
import json
import time
import threading
import traceback
import websocket
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import deque

import ccxt
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

# ============================================================
# CONFIG
# ============================================================
SYMBOL_MEXC = "SILVER/USDT:USDT"  # Silver perp on MEXC (works from Render US)
SYMBOL_BINANCE = "XAG/USDT"       # Silver on Binance (works from India)
SYMBOL_DISPLAY = "SILVER/USDT (Silver)"
TIMEFRAME = "1m"
CAPITAL = 100.0        # INR
LEVERAGE = 50
RISK_PCT = 0.50        # 50% risk per trade
TP_ATR = 1.8
SL_ATR = 0.6
CANDLE_BUFFER = 200    # Keep last 200 candles for indicators
FETCH_INTERVAL = 10    # Fetch candles every 10 seconds
WS_TICK_INTERVAL = 1   # WebSocket price tick every 1 second
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005

app = Flask(__name__)

# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class PaperTrade:
    id: int
    symbol: str
    side: str           # 'long' or 'short'
    entry_price: float
    size: float         # notional in INR
    stop_loss: float
    take_profit: float
    entry_time: str
    exit_price: float = None
    exit_time: str = None
    exit_reason: str = None
    pnl: float = None
    pnl_pct: float = None
    status: str = "open"

# ============================================================
# STRATEGY ENGINE (inline, no imports needed)
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    v = df["volume"].values
    n = len(df)

    # EMA (seeded with first value — matches backtester, never NaN)
    def ema(data, period):
        data = np.asarray(data, dtype=np.float64)
        result = np.empty(len(data), dtype=np.float64)
        if len(data) == 0:
            return result
        alpha = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    df["ema_fast"] = ema(c, 8)
    df["ema_slow"] = ema(c, 21)

    # ATR
    tr = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    tr[0] = h[0] - lo[0]
    atr = ema(tr, 14)
    df["atr_raw"] = atr
    df["atr_smooth"] = ema(atr, 14)

    # RSI
    deltas = np.diff(c, prepend=c[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = ema(gains, 14)
    avg_loss = ema(losses, 14)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    df["bb_middle"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # ADX
    plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0)
    minus_dm = np.maximum(-np.diff(lo, prepend=lo[0]), 0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    atr14 = ema(tr, 14)
    plus_di = 100 * ema(plus_dm, 14) / np.where(atr14 > 0, atr14, 1)
    minus_di = 100 * ema(minus_dm, 14) / np.where(atr14 > 0, atr14, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    df["adx"] = ema(dx, 14)

    # Momentum (plain numpy array — no pandas chained assignment)
    mom_arr = c - np.roll(c, 10)
    mom_arr[0:10] = 0.0
    df["momentum"] = mom_arr

    # Volume spike
    vol_sma = pd.Series(v).rolling(20).mean().values
    df["vol_spike"] = np.where(vol_sma > 0, v / vol_sma, 1.0)

    # Volume imbalance (bullish volume ratio)
    bull_vol = np.where(c > o, v, 0)
    bear_vol = np.where(c < o, v, 0)
    rolling_bull = pd.Series(bull_vol).rolling(20).sum().values
    rolling_bear = pd.Series(bear_vol).rolling(20).sum().values
    total = rolling_bull + rolling_bear
    df["vol_imbalance"] = np.where(total > 0, rolling_bull / total, 0.5)

    # Z-score
    mean20 = pd.Series(c).rolling(20).mean().values
    std20_z = pd.Series(c).rolling(20).std().values
    df["zscore"] = np.where(std20_z > 0, (c - mean20) / std20_z, 0)

    return df


def generate_breakout_signals(df: pd.DataFrame) -> np.ndarray:
    """Breakout strategy signals."""
    n = len(df)
    sig = np.zeros(n, dtype=np.int32)
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    v = df["volume"].values
    ema_f = df["ema_fast"].values
    ema_s = df["ema_slow"].values
    bb_u = df["bb_upper"].values
    bb_l = df["bb_lower"].values
    bb_m = df["bb_middle"].values
    mom = df["momentum"].values
    atr_v = df["atr_smooth"].values

    ls = -999; dc = 0; cd = None; lb = 20
    for i in range(lb + 10, n):
        ts = pd.Timestamp(df.index[i])
        day = ts.date()
        if day != cd: cd = day; dc = 0
        if dc >= 10 or i - ls < 10: continue

        p = c[i]; a = atr_v[i]
        if a / p < 0.0001 or np.isnan(a): continue
        if np.isnan(bb_m[i]) or bb_m[i] <= 0: continue

        rh = np.max(h[i-lb:i]); rl = np.min(lo[i-lb:i])
        bw = (bb_u[i] - bb_l[i]) / bb_m[i]
        av = np.mean(v[max(0,i-50):i]) if i >= 50 else np.mean(v[:i])
        vr = v[i] / av if av > 0 else 1
        rng_pct = (rh - rl) / p if p > 0 else 0
        consol = rng_pct < 0.006 and bw < 0.008

        if p > rh:
            pts = 0
            bo = (p - rh) / (rh - rl) if (rh - rl) > 0 else 0
            if bo > 0.3: pts += 3
            elif bo > 0.1: pts += 2
            else: pts += 1
            if vr > 2.5: pts += 2
            elif vr > 1.8: pts += 1
            if consol: pts += 2
            if not np.isnan(ema_f[i]) and ema_f[i] > ema_s[i]: pts += 1
            if not np.isnan(mom[i]) and mom[i] > 0: pts += 1
            if c[i] > o[i]: pts += 0.5
            if pts >= 5.5:
                sig[i] = 1; ls = i; dc += 1; continue

        if p < rl:
            pts = 0
            bo = (rl - p) / (rh - rl) if (rh - rl) > 0 else 0
            if bo > 0.3: pts += 3
            elif bo > 0.1: pts += 2
            else: pts += 1
            if vr > 2.5: pts += 2
            elif vr > 1.8: pts += 1
            if consol: pts += 2
            if not np.isnan(ema_f[i]) and ema_f[i] < ema_s[i]: pts += 1
            if not np.isnan(mom[i]) and mom[i] < 0: pts += 1
            if c[i] < o[i]: pts += 0.5
            if pts >= 5.5:
                sig[i] = -1; ls = i; dc += 1

    return sig


def generate_momentum_signals(df: pd.DataFrame) -> np.ndarray:
    """Momentum strategy signals."""
    n = len(df)
    sig = np.zeros(n, dtype=np.int32)
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    o = df["open"].values
    ema_f = df["ema_fast"].values
    ema_s = df["ema_slow"].values
    vol_spike = df["vol_spike"].values
    vol_imb = df["vol_imbalance"].values
    adx_v = df["adx"].values

    ls = -999; dc = 0; cd = None
    for i in range(30, n):
        ts = pd.Timestamp(df.index[i])
        day = ts.date()
        if day != cd: cd = day; dc = 0
        if dc >= 5 or i - ls < 20: continue

        a = df["atr_smooth"].values[i]
        price = c[i]
        if np.isnan(a) or a / price < 0.0002: continue

        rng = h[i] - lo[i]
        if rng <= 0: continue
        body = c[i] - o[i]
        body_pct = abs(body) / rng
        size_pct = rng / price
        if size_pct < 0.0003: continue

        # BULL
        bp = 0.0
        if body > 0 and body_pct > 0.65:
            bp += 2.5 if size_pct > 0.001 else 1.5
        if vol_spike[i] > 2.5: bp += 2.0
        elif vol_spike[i] > 1.8: bp += 1.5
        elif vol_spike[i] > 1.3: bp += 0.5
        if not np.isnan(ema_f[i]) and c[i] > ema_f[i] > ema_s[i]: bp += 1.5
        elif not np.isnan(ema_f[i]) and c[i] > ema_f[i]: bp += 0.5
        if i >= 5 and h[i] > np.max(h[i-5:i]): bp += 1.0
        if vol_imb[i] > 0.60: bp += 0.5
        if not np.isnan(adx_v[i]) and adx_v[i] > 20: bp += 0.5
        if i >= 1 and c[i-1] > o[i-1] and body > 0: bp += 0.5

        # BEAR
        brp = 0.0
        if body < 0 and body_pct > 0.65:
            brp += 2.5 if size_pct > 0.001 else 1.5
        if vol_spike[i] > 2.5: brp += 2.0
        elif vol_spike[i] > 1.8: brp += 1.5
        elif vol_spike[i] > 1.3: brp += 0.5
        if not np.isnan(ema_f[i]) and c[i] < ema_f[i] < ema_s[i]: brp += 1.5
        elif not np.isnan(ema_f[i]) and c[i] < ema_f[i]: brp += 0.5
        if i >= 5 and lo[i] < np.min(lo[i-5:i]): brp += 1.0
        if vol_imb[i] < 0.40: brp += 0.5
        if not np.isnan(adx_v[i]) and adx_v[i] > 20: brp += 0.5
        if i >= 1 and c[i-1] < o[i-1] and body < 0: brp += 0.5

        if bp >= 4.5 and bp > brp + 1:
            sig[i] = 1; ls = i; dc += 1
        elif brp >= 4.5 and brp > bp + 1:
            sig[i] = -1; ls = i; dc += 1

    return sig


def generate_combined_signals(df: pd.DataFrame) -> np.ndarray:
    """Combined Breakout + Momentum signals."""
    sig_bo = generate_breakout_signals(df)
    sig_mo = generate_momentum_signals(df)
    combined = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        if sig_bo[i] == 1 and sig_mo[i] == 1:
            combined[i] = 1
        elif sig_bo[i] == -1 and sig_mo[i] == -1:
            combined[i] = -1
    return combined


# ============================================================
# LIVE TRADING ENGINE
# ============================================================
class PaperTradingEngine:
    def __init__(self):
        self.exchange = None
        self.exchange_name = None
        # Try exchanges in order: MEXC works everywhere (incl Render US)
        # Binance/Bybit only work from India
        exchanges_to_try = [
            ('mexc', ccxt.mexc, {'enableRateLimit': True, 'timeout': 15000, 'options': {'defaultType': 'swap'}}),
            ('binance', ccxt.binance, {'enableRateLimit': True, 'timeout': 15000, 'options': {'defaultType': 'future'}}),
            ('bybit', ccxt.bybit, {'enableRateLimit': True, 'timeout': 15000, 'options': {'defaultType': 'linear'}}),
        ]
        for name, cls, opts in exchanges_to_try:
            try:
                ex = cls(opts)
                sym = SYMBOL_MEXC if name == 'mexc' else SYMBOL_BINANCE
                ex.fetch_ohlcv(sym, '1m', limit=2)
                self.exchange = ex
                self.exchange_name = name
                print(f"Using exchange: {name} | {sym}")
                break
            except Exception as e:
                print(f"{name} failed: {e}")
                continue
        if self.exchange is None:
            raise RuntimeError("No exchange available")
        self._warm = False
        self.candles = deque(maxlen=CANDLE_BUFFER)
        self.trades = []
        self.trade_id = 0
        self.position = None
        self.capital = CAPITAL
        self.equity_curve = []
        self.last_signal_bar = -999
        self.running = False
        self.last_fetch = None
        self.last_tick = None
        self.tick_count = 0
        self.last_diag = {}
        self._tick_lock = threading.Lock()
        self.error = None
        self.live_price = None          # real-time price from WebSocket
        self.live_price_time = None     # timestamp of last price update
        self.ws_connected = False
        self.state_file = Path(__file__).parent / "paper_state.json"
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.capital = state.get('capital', CAPITAL)
                self.trades = [PaperTrade(**t) for t in state.get('trades', [])]
                self.trade_id = state.get('trade_id', 0)
                self.position = PaperTrade(**state['position']) if state.get('position') else None
                self.equity_curve = state.get('equity_curve', [])
            except Exception:
                pass

    def _save_state(self):
        state = {
            'capital': self.capital,
            'trades': [asdict(t) for t in self.trades[-100:]],
            'trade_id': self.trade_id,
            'position': asdict(self.position) if self.position else None,
            'equity_curve': self.equity_curve[-500:],
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def fetch_candles(self):
        """Fetch latest candles from exchange."""
        try:
            symbol = SYMBOL_MEXC if self.exchange_name == 'mexc' else SYMBOL_BINANCE
            ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_BUFFER)
            self.candles.clear()
            for c in ohlcv:
                self.candles.append({
                    'timestamp': pd.Timestamp(c[0], unit='ms'),
                    'open': c[1],
                    'high': c[2],
                    'low': c[3],
                    'close': c[4],
                    'volume': c[5],
                })
            self.last_fetch = datetime.now(timezone.utc).isoformat()
            self.error = None
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def run_strategy(self):
        """Run strategy on current candle buffer (uses last CLOSED bar)."""
        if len(self.candles) < 60:
            self.last_diag = {'reason': 'warming_up', 'candles': len(self.candles)}
            return None

        df = pd.DataFrame(list(self.candles))
        df.set_index('timestamp', inplace=True)
        df = compute_indicators(df)
        sig_bo = generate_breakout_signals(df)
        sig_mo = generate_momentum_signals(df)
        signals = generate_combined_signals(df)

        # Last CLOSED 1m bar — never trade the still-forming candle
        closed_idx = len(df) - 2
        price = float(df['close'].iloc[closed_idx])
        atr_v = float(df['atr_smooth'].iloc[closed_idx])
        diag = {
            'bar_time': str(df.index[closed_idx]),
            'close': round(price, 4),
            'atr': round(atr_v, 6) if atr_v == atr_v else None,
            'atr_price_ratio': round(atr_v / price, 6) if atr_v == atr_v and price else None,
            'breakout_votes_total': int(np.sum(sig_bo != 0)),
            'momentum_votes_total': int(np.sum(sig_mo != 0)),
            'combined_votes_total': int(np.sum(signals != 0)),
            'closed_bar_signal': int(signals[closed_idx]),
        }
        if signals[closed_idx] == 0:
            diag['reason'] = 'no_signal_on_closed_bar'
            self.last_diag = diag
            return None
        if closed_idx - self.last_signal_bar < 10:
            diag['reason'] = 'cooldown'
            self.last_diag = diag
            return None
        if not (atr_v == atr_v) or atr_v <= 0:
            diag['reason'] = 'bad_atr'
            self.last_diag = diag
            return None

        diag['reason'] = 'signal'
        self.last_diag = diag
        return {
            'direction': 'long' if signals[closed_idx] == 1 else 'short',
            'price': price,
            'atr': atr_v,
            'bar_idx': closed_idx,
        }

    def open_trade(self, signal):
        """Open a paper trade."""
        if self.position is not None:
            return None

        direction = signal['direction']
        price = signal['price']
        atr = signal['atr']

        if np.isnan(atr) or atr <= 0:
            return None

        tp_dist = TP_ATR * atr
        sl_dist = SL_ATR * atr

        if direction == 'long':
            tp = price + tp_dist
            sl = price - sl_dist
        else:
            tp = price - tp_dist
            sl = price + sl_dist

        # Position size
        risk_per_unit = sl_dist
        risk_amount = self.capital * RISK_PCT
        notional = risk_amount / (risk_per_unit / price) if risk_per_unit > 0 else 0
        max_notional = self.capital * LEVERAGE
        notional = min(notional, max_notional)
        notional = max(notional, 0)

        if notional < 1:
            return None

        self.trade_id += 1
        trade = PaperTrade(
            id=self.trade_id,
            symbol=SYMBOL_DISPLAY,
            side=direction,
            entry_price=price,
            size=notional,
            stop_loss=sl,
            take_profit=tp,
            entry_time=datetime.now(timezone.utc).isoformat(),
        )
        self.position = trade
        self.last_signal_bar = signal['bar_idx']
        self._save_state()
        print(f"OPEN {direction} @ {price:.4f} | SL {sl:.4f} TP {tp:.4f} | notional {notional:.2f} INR", flush=True)
        return trade

    def check_exit(self):
        """Check if current position should be closed."""
        if self.position is None or len(self.candles) == 0:
            return

        pos = self.position
        candle = self.candles[-1]
        price = candle['close']
        high = candle['high']
        low = candle['low']

        exit_price = None
        exit_reason = None

        if pos.side == 'long':
            if low <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = 'stop_loss'
            elif high >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = 'take_profit'
        else:
            if high >= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = 'stop_loss'
            elif low <= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = 'take_profit'

        if exit_price is not None:
            self.close_trade(exit_price, exit_reason)

    def close_trade(self, exit_price, reason):
        """Close the current position."""
        pos = self.position
        if pos is None:
            return

        if pos.side == 'long':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_pct *= LEVERAGE
        pnl_inr = pnl_pct * self.capital
        fee = TAKER_FEE * 2 * self.capital * LEVERAGE
        pnl_net = pnl_inr - fee

        pos.exit_price = exit_price
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.exit_reason = reason
        pos.pnl = round(pnl_net, 4)
        pos.pnl_pct = round(pnl_pct * 100, 2)
        pos.status = "closed"

        self.capital += pnl_net
        self.trades.append(pos)
        self.position = None
        self.equity_curve.append({
            'time': pos.exit_time,
            'equity': round(self.capital, 2),
        })
        self._save_state()
        print(f"CLOSED {reason} @ {exit_price:.4f} | pnl {pos.pnl} INR | capital {self.capital:.2f}", flush=True)

    def maybe_tick(self, min_interval=FETCH_INTERVAL):
        """Tick at most once per min_interval. Safe to call from any thread
        (request handlers, background loop). Non-blocking if a tick is running."""
        if not self.running:
            return False
        if self.last_tick:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(self.last_tick)).total_seconds()
                if age < min_interval:
                    return False
            except Exception:
                pass
        if not self._tick_lock.acquire(blocking=False):
            return False
        try:
            self.tick()
        finally:
            self._tick_lock.release()
        return True

    def tick(self):
        """One iteration: fetch, check, maybe trade."""
        if not self.running:
            return
        self.tick_count += 1
        self.last_tick = datetime.now(timezone.utc).isoformat()
        try:
            self.fetch_candles()
            if self.position:
                self.check_exit()
            else:
                signal = self.run_strategy()
                if signal:
                    self.open_trade(signal)
        except Exception as e:
            self.error = f"Tick error: {e}\n{traceback.format_exc()}"

    def get_status(self):
        """Get full status dict."""
        win_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        lose_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)

        return {
            'symbol': SYMBOL_DISPLAY,
            'exchange': getattr(self, 'exchange_name', 'unknown'),
            'capital': round(self.capital, 2),
            'starting_capital': CAPITAL,
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round((self.capital - CAPITAL) / CAPITAL * 100, 2),
            'total_trades': len(self.trades),
            'win_trades': len(win_trades),
            'lose_trades': len(lose_trades),
            'win_rate': round(len(win_trades) / len(self.trades) * 100, 1) if self.trades else 0,
            'avg_win': round(np.mean([t.pnl for t in win_trades]), 2) if win_trades else 0,
            'avg_loss': round(np.mean([t.pnl for t in lose_trades]), 2) if lose_trades else 0,
            'position': asdict(self.position) if self.position else None,
            'last_fetch': self.last_fetch,
            'last_tick': self.last_tick,
            'tick_count': self.tick_count,
            'last_diag': self.last_diag,
            'live_price': self.live_price,
            'live_price_time': self.live_price_time,
            'ws_connected': self.ws_connected,
            'error': self.error,
            'running': self.running,
            'candles_loaded': len(self.candles),
            'config': {
                'tp_atr': TP_ATR,
                'sl_atr': SL_ATR,
                'risk_pct': RISK_PCT,
                'leverage': LEVERAGE,
            },
        }

    def get_trades(self, limit=50):
        """Get recent trades."""
        recent = self.trades[-limit:]
        return [asdict(t) for t in reversed(recent)]

    def get_equity_curve(self):
        return self.equity_curve[-200:]

    def start(self):
        self.running = True
        self._save_state()

    def stop(self):
        self.running = False
        self._save_state()


# ============================================================
# GLOBAL STATE
# ============================================================
engine = PaperTradingEngine()


# ============================================================
# LIVE PRICE POLLER (REST ticker every 3s — reliable, no WS guesswork)
# ============================================================
def live_price_loop():
    """Poll exchange ticker for real-time price (drives 1s TP/SL checks)."""
    while True:
        try:
            if engine.exchange is not None:
                symbol = SYMBOL_MEXC if engine.exchange_name == 'mexc' else SYMBOL_BINANCE
                t = engine.exchange.fetch_ticker(symbol)
                px = t.get('last') or t.get('close')
                if px:
                    engine.live_price = float(px)
                    engine.live_price_time = datetime.now(timezone.utc).isoformat()
                    engine.ws_connected = True
        except Exception:
            try:
                if engine.live_price_time:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(engine.live_price_time)).total_seconds()
                    if age > 15:
                        engine.ws_connected = False
            except Exception:
                pass
        time.sleep(3)


# ============================================================
# FAST TP/SL CHECK (runs every 1 second on live price)
# ============================================================
def fast_exit_loop():
    """Check TP/SL every 1 second using real-time WebSocket price."""
    while True:
        try:
            if engine.running and engine.position and engine.live_price:
                pos = engine.position
                price = engine.live_price
                exit_price = None
                exit_reason = None
                
                if pos.side == 'long':
                    # Need high/low — use live_price as both for real-time
                    # Check against SL first (worst case), then TP
                    if price <= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = 'stop_loss'
                    elif price >= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = 'take_profit'
                else:
                    if price >= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = 'stop_loss'
                    elif price <= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = 'take_profit'
                
                if exit_price is not None:
                    engine.close_trade(exit_price, exit_reason)
        except Exception:
            pass
        time.sleep(WS_TICK_INTERVAL)


# ============================================================
# CANDLE FETCH LOOP (every 10 seconds)
# ============================================================
def background_loop():
    """Background thread: fetch candles + run strategy. Never dies.
    (Backup path — requests also drive ticks via maybe_tick.)"""
    while True:
        try:
            engine.maybe_tick()
        except Exception as e:
            try:
                engine.error = f"Loop error: {e}"
            except Exception:
                pass
        time.sleep(FETCH_INTERVAL)


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    # Drive the engine from traffic: works even if background threads stall.
    # Dashboard polls this every 1s, so ticks stay on ~10s cadence.
    try:
        engine.maybe_tick()
    except Exception:
        pass
    return jsonify(engine.get_status())


@app.route('/api/tick', methods=['GET', 'POST'])
def api_tick():
    """External cron/keep-alive hook: force an engine tick."""
    try:
        ran = engine.maybe_tick(min_interval=5)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})
    return jsonify({'ok': True, 'ticked': ran, 'tick_count': engine.tick_count})

@app.route('/api/trades')
def api_trades():
    try:
        limit = int(request.args.get('limit', 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))
    return jsonify(engine.get_trades(limit=limit))

@app.route('/api/equity')
def api_equity():
    return jsonify(engine.get_equity_curve())

@app.route('/api/start', methods=['POST'])
def api_start():
    engine.start()
    return jsonify({'ok': True})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    engine.stop()
    return jsonify({'ok': True})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    engine.capital = CAPITAL
    engine.trades = []
    engine.position = None
    engine.equity_curve = []
    engine.trade_id = 0
    engine._save_state()
    return jsonify({'ok': True})


# ============================================================
# STARTUP (runs when gunicorn loads the module)
# ============================================================
def _start_background():
    # Thread 1: Candle fetch + strategy (every 10s)
    t1 = threading.Thread(target=background_loop, daemon=True)
    t1.start()

    # Thread 2: Live price poller (REST ticker every 3s)
    t2 = threading.Thread(target=live_price_loop, daemon=True)
    t2.start()

    # Thread 3: Fast TP/SL exit check (every 1s)
    t3 = threading.Thread(target=fast_exit_loop, daemon=True)
    t3.start()

    engine.fetch_candles()
    engine.running = True
    print("Started: candle loop (10s) + live price poller (3s) + fast exit (1s)")

_start_background()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
