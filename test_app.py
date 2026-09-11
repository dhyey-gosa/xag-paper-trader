from app import PaperTradingEngine, compute_indicators, generate_combined_signals
import pandas as pd

engine = PaperTradingEngine()
print("Fetching candles...")
ok = engine.fetch_candles()
print(f"Fetched: {ok}, candles: {len(engine.candles)}")

if ok:
    df = pd.DataFrame(list(engine.candles))
    df.set_index("timestamp", inplace=True)
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Latest price: {df.close.iloc[-1]}")
    
    df = compute_indicators(df)
    signals = generate_combined_signals(df)
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    print(f"Signals: {n_long} long, {n_short} short")
    
    last_sig = signals[-1]
    if last_sig != 0:
        direction = "LONG" if last_sig == 1 else "SHORT"
        print(f"ACTIVE SIGNAL: {direction} @ {df.close.iloc[-1]}")
    else:
        print("No signal on last bar")
    
    # Test strategy run
    signal = engine.run_strategy()
    if signal:
        print(f"Strategy signal: {signal['direction']} @ {signal['price']}")
    else:
        print("No strategy signal (need more data or cooldown)")
