from app import PaperTradingEngine

engine = PaperTradingEngine()
print(f"Exchange: {engine.exchange_name}")
ok = engine.fetch_candles()
print(f"Fetched: {ok}, candles: {len(engine.candles)}")

if ok:
    c = list(engine.candles)[-1]
    print(f"Latest: {c['timestamp']} close={c['close']} vol={c['volume']}")
    
    signal = engine.run_strategy()
    if signal:
        d = signal["direction"]
        p = signal["price"]
        a = signal["atr"]
        print(f"SIGNAL: {d} @ {p} ATR={a:.4f}")
    else:
        print("No signal (need more data or cooldown)")
