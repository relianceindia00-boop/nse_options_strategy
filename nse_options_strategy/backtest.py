import pandas as pd
import numpy as np

def run_backtest(df_with_signals, stop_loss_pct=0.002, take_profit_pct=0.005):
    """
    Simulates trades based on CE/PE signals.
    Because we don't have historical option prices, we proxy option PnL 
    using the underlying asset's percentage move.
    
    stop_loss_pct: 0.2% underlying move ~ 10-15% option premium move (Delta dependent)
    take_profit_pct: 0.5% underlying move ~ 25-40% option premium move
    """
    
    trades = []
    in_trade = False
    entry_price = 0
    trade_type = None # 'CE' or 'PE'
    entry_time = None
    
    # We will iterate through rows for a simple backtest
    # This can be slow for huge datasets, but fine for 5-day 5m data
    
    for time, row in df_with_signals.iterrows():
        # Check for exits if in trade
        if in_trade:
            # Calculate current move
            if trade_type == 'CE':
                pnl_pct = (row['Close'] - entry_price) / entry_price
            else: # PE
                pnl_pct = (entry_price - row['Close']) / entry_price
                
            # Exit conditions
            hit_sl = pnl_pct <= -stop_loss_pct
            hit_tp = pnl_pct >= take_profit_pct
            
            # Also exit at end of day (3:15 PM / 15:15) to avoid overnight gap risks (pure intraday options buying)
            eod_exit = time.hour == 15 and time.minute >= 15
            
            if hit_sl or hit_tp or eod_exit:
                exit_price = row['Close']
                trades.append({
                    'Entry_Time': entry_time,
                    'Exit_Time': time,
                    'Type': trade_type,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Pct': pnl_pct * 100, # as percentage
                    'Reason': 'SL' if hit_sl else ('TP' if hit_tp else 'EOD')
                })
                in_trade = False
                
        # Check for entries if not in trade
        if not in_trade:
            # Don't take new trades after 3:00 PM
            if time.hour == 15:
                continue
                
            if row['CE_Signal']:
                in_trade = True
                trade_type = 'CE'
                entry_price = row['Close']
                entry_time = time
            elif row['PE_Signal']:
                in_trade = True
                trade_type = 'PE'
                entry_price = row['Close']
                entry_time = time
                
    trades_df = pd.DataFrame(trades)
    
    # Compute summary stats
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL_Pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = trades_df['PnL_Pct'].sum()
        max_drawdown = trades_df['PnL_Pct'].cumsum().min() # very rough estimate
    else:
        total_trades = 0
        win_rate = 0.0
        total_pnl = 0.0
        max_drawdown = 0.0
        
    summary = {
        'Total Trades': total_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Total Returns (Underlying % sum)': round(total_pnl, 2),
        'Max Drawdown (%)': round(max_drawdown, 2)
    }
    
    return trades_df, summary

if __name__ == "__main__":
    from data_loader import fetch_data
    from indicators import apply_indicators
    from strategies import generate_signals
    
    df = fetch_data()
    df = apply_indicators(df)
    df = generate_signals(df, setup_number=1)
    
    trades, summary = run_backtest(df)
    print("\nBacktest Summary for Setup 1:")
    for k, v in summary.items():
        print(f"{k}: {v}")
        
    if not trades.empty:
        print("\nFirst few trades:")
        print(trades.head())
