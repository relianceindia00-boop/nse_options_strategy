import pandas as pd
import pandas_ta as ta

def apply_indicators(df):
    """
    Applies technical indicators to the DataFrame using pandas_ta.
    Requires columns: Open, High, Low, Close, Volume.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # If volume is 0 everywhere (sometimes true for index in yfinance), 
    # we need a synthetic volume for VWAP/VWMA to work without NaNs.
    if df['Volume'].sum() == 0:
        df['Volume'] = 1000  # Give dummy volume to allow VWAP to just track price

    # 1. Stochastics (34, 5, 21) -> Length=34, %K=5, %D=21
    # pandas_ta stochastic returns stoch_k, stoch_d
    stoch = df.ta.stoch(high='High', low='Low', close='Close', k=34, d=21, smooth_k=5)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
        # Rename to generic names for easy access
        df.rename(columns={stoch.columns[0]: 'STOCH_k', stoch.columns[1]: 'STOCH_d'}, inplace=True)
    else:
        df['STOCH_k'] = 50
        df['STOCH_d'] = 50

    # 2. VWAP
    # VWAP usually resets daily, we can use pandas_ta vwap
    vwap = df.ta.vwap(high='High', low='Low', close='Close', volume='Volume')
    if vwap is not None:
        df['VWAP'] = vwap
    else:
        df['VWAP'] = df['Close']

    # 3. VWMA (Volume Weighted Moving Average)
    vwma = df.ta.vwma(close='Close', volume='Volume', length=20)
    if vwma is not None:
        df['VWMA'] = vwma
    else:
        df['VWMA'] = df['Close']

    # 4. ADX (Length 14)
    adx = df.ta.adx(high='High', low='Low', close='Close', length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
        df.rename(columns={adx.columns[0]: 'ADX'}, inplace=True)
    else:
        df['ADX'] = 0

    # 5. EMA 9 and 21
    df['EMA_9'] = df.ta.ema(close='Close', length=9)
    df['EMA_21'] = df.ta.ema(close='Close', length=21)

    # 6. Supertrend (Length 10, Multiplier 3)
    st = df.ta.supertrend(high='High', low='Low', close='Close', length=10, multiplier=3.0)
    if st is not None:
        df = pd.concat([df, st], axis=1)
        # SUPERT_10_3.0 is usually the trend line, SUPERTd_10_3.0 is the direction (1 or -1)
        st_dir_col = [c for c in st.columns if 'SUPERTd' in c][0]
        df.rename(columns={st_dir_col: 'Supertrend_Dir'}, inplace=True)
    else:
        df['Supertrend_Dir'] = 1

    # 7. RSI (Length 14)
    df['RSI'] = df.ta.rsi(close='Close', length=14)

    # 8. Bollinger Bands (Length 20, StdDev 2)
    bb = df.ta.bbands(close='Close', length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        # Rename to Upper, Mid, Lower
        lower_col = [c for c in bb.columns if 'BBL' in c][0]
        mid_col = [c for c in bb.columns if 'BBM' in c][0]
        upper_col = [c for c in bb.columns if 'BBU' in c][0]
        bandwidth_col = [c for c in bb.columns if 'BBB' in c][0]
        
        df.rename(columns={
            lower_col: 'BB_Lower',
            mid_col: 'BB_Mid',
            upper_col: 'BB_Upper',
            bandwidth_col: 'BB_Bandwidth'
        }, inplace=True)
    else:
        df['BB_Lower'] = df['Close']
        df['BB_Upper'] = df['Close']
        df['BB_Bandwidth'] = 0

    # 9. Candlestick Patterns
    
    # Inside Bar
    # An inside bar is where High < Prev High and Low > Prev Low
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Is_Inside_Bar'] = (df['High'] < df['Prev_High']) & (df['Low'] > df['Prev_Low'])
    
    # NR7 (Narrowest Range in 7 periods)
    df['Range'] = df['High'] - df['Low']
    # min range over last 7 periods (including current)
    df['Min_Range_7'] = df['Range'].rolling(7).min()
    # NR7 is True if current range is equal to the minimum range of the last 7
    df['Is_NR7'] = (df['Range'] == df['Min_Range_7']) & (df['Range'] > 0)
    
    # Previous Day High/Low (Daily timeframe equivalent)
    # Group by date to find daily highs and lows, shift by 1 to get previous day
    # Assuming index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        daily_data = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'}).shift(1)
        daily_data.columns = ['PDH', 'PDL']
        
        # Map PDH and PDL back to the intraday dataframe based on date
        df['Date_Only'] = df.index.date
        df = df.merge(daily_data, left_on='Date_Only', right_index=True, how='left')
        df.drop(columns=['Date_Only'], inplace=True)
    else:
        df['PDH'] = df['High'].rolling(75).max().shift(1) # Approximation if not datetime
        df['PDL'] = df['Low'].rolling(75).min().shift(1)

    return df

if __name__ == "__main__":
    from data_loader import fetch_data
    df = fetch_data()
    df_with_indicators = apply_indicators(df)
    print("Columns after indicators:")
    print(df_with_indicators.columns.tolist())
    print("\nSample Data:")
    print(df_with_indicators[['Close', 'STOCH_k', 'VWAP', 'ADX', 'Supertrend_Dir', 'Is_Inside_Bar', 'PDH']].tail())

