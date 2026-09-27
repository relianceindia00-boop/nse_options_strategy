from data_loader import fetch_data
from indicators import apply_indicators
from strategies import generate_signals
from backtest import run_backtest

df = fetch_data(symbol="^NSEI", period="1mo", interval="5m")
df = apply_indicators(df)

for setup_id in [1, 3, 15]:
    df_sig = generate_signals(df, setup_number=setup_id)
    trades, summary = run_backtest(df_sig)
    print(f"--- Setup {setup_id} ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print()
