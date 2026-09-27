import pandas as pd

def apply_indicators(df):
    """
    Applies technical indicators to the DataFrame using standard pandas (since pandas-ta fails on python 3.14).
    Requires columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()

    if df['Volume'].sum() == 0:
        df['Volume'] = 1000  

    # 1. Stochastics (34, 5, 21) -> Length=34, %K=5, %D=21
    df['L14'] = df['Low'].rolling(window=34).min()
    df['H14'] = df['High'].rolling(window=34).max()
    df['STOCH_k'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['STOCH_k'] = df['STOCH_k'].rolling(window=5).mean()
    df['STOCH_d'] = df['STOCH_k'].rolling(window=21).mean()
    df.drop(['L14', 'H14'], axis=1, inplace=True)

    # 2. VWAP
    q = df['Volume']
    p = df['Close']
    df['VWAP'] = (p * q).cumsum() / q.cumsum()

    # 3. VWMA (Volume Weighted Moving Average)
    df['VWMA'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # 4. ADX (Length 14)
    df['ADX'] = 0 # Dummy ADX for now

    # 5. EMA 9 and 21
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # 6. Supertrend (Length 10, Multiplier 3)
    df['Supertrend_Dir'] = 1 # Dummy supertrend direction

    # 7. RSI (Length 14)
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(com=14 - 1, adjust=False).mean()
    roll_down = down.abs().ewm(com=14 - 1, adjust=False).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # 8. Bollinger Bands (Length 20, StdDev 2)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * 2)
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

    # 9. Candlestick Patterns
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Is_Inside_Bar'] = (df['High'] < df['Prev_High']) & (df['Low'] > df['Prev_Low'])
    
    df['Range'] = df['High'] - df['Low']
    df['Min_Range_7'] = df['Range'].rolling(7).min()
    df['Is_NR7'] = (df['Range'] == df['Min_Range_7']) & (df['Range'] > 0)
    
    if isinstance(df.index, pd.DatetimeIndex):
        daily_data = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'}).shift(1)
        daily_data.columns = ['PDH', 'PDL']
        df['Date_Only'] = df.index.date
        df = df.merge(daily_data, left_on='Date_Only', right_index=True, how='left')
        df.drop(columns=['Date_Only'], inplace=True)
    else:
        df['PDH'] = df['High'].rolling(75).max().shift(1) 
        df['PDL'] = df['Low'].rolling(75).min().shift(1)

    return df
