import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(symbol="^NSEI", period="5d", interval="5m"):
    """
    Fetch historical data from Yahoo Finance.
    For NSE indices, typically use ^NSEI (Nifty 50) or ^NSEBANK (Bank Nifty).
    """
    print(f"Fetching {period} data for {symbol} at {interval} interval...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        print(f"Warning: No data fetched for {symbol}. Try checking the symbol or interval.")
        return df

    # Flatten MultiIndex columns if present (yfinance sometimes returns them for single symbols)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.index = pd.to_datetime(df.index)
    
    # If the index does not have a timezone, we localize it (if possible)
    if df.index.tzinfo is None:
        try:
            df.index = df.index.tz_localize('Asia/Kolkata')
        except Exception:
            pass
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')

    # Ensure required columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            df[col] = 0.0

    # Generate synthetic Open Interest (OI)
    # Since real OI is not provided by yfinance, we simulate it based on volume and a random walk
    np.random.seed(42)  # For reproducibility
    base_oi = 1000000
    oi_changes = (np.random.randn(len(df)) * df['Volume'].mean() * 0.1).astype(int)
    # Add some correlation with volume spikes
    volume_spike_mask = df['Volume'] > df['Volume'].rolling(10).mean() * 1.5
    oi_changes = np.where(volume_spike_mask, np.abs(oi_changes) * 1.5, oi_changes)
    
    synthetic_oi = base_oi + np.cumsum(oi_changes)
    df['OI'] = synthetic_oi

    return df

if __name__ == "__main__":
    data = fetch_data()
    print("Data head:")
    print(data.head())
    print(f"Total rows: {len(data)}")
    print("Columns:", data.columns)
