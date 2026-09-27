import pandas as pd
import numpy as np
from indicators import apply_indicators

def generate_signals(df, setup_number):
    """
    Applies the chosen strategy setup and returns a DataFrame with 'CE_Signal' and 'PE_Signal' boolean columns.
    """
    # Ensure indicators are applied
    if 'STOCH_k' not in df.columns:
        df = apply_indicators(df)
        
    df = df.copy()
    
    # Initialize signals to False
    df['CE_Signal'] = False
    df['PE_Signal'] = False
    
    # Common conditions (to detect crosses)
    df['Prev_STOCH_k'] = df['STOCH_k'].shift(1)
    df['Prev_ADX'] = df['ADX'].shift(1)
    df['Prev_RSI'] = df['RSI'].shift(1)
    
    stoch_cross_60_up = (df['Prev_STOCH_k'] <= 60) & (df['STOCH_k'] > 60)
    stoch_cross_40_down = (df['Prev_STOCH_k'] >= 40) & (df['STOCH_k'] < 40)
    
    adx_cross_22_up = (df['Prev_ADX'] <= 22) & (df['ADX'] > 22)
    
    # 1. Stoch + VWAP Breakout
    if setup_number == 1:
        df['CE_Signal'] = stoch_cross_60_up & (df['Close'] > df['VWAP'])
        df['PE_Signal'] = stoch_cross_40_down & (df['Close'] < df['VWAP'])

    # 2. Stoch + VWMA + VWAP
    elif setup_number == 2:
        df['CE_Signal'] = (df['STOCH_k'] > 60) & (df['Close'] > df['VWMA']) & (df['Close'] > df['VWAP'])
        df['PE_Signal'] = (df['STOCH_k'] < 40) & (df['Close'] < df['VWMA']) & (df['Close'] < df['VWAP'])

    # 3. ADX Momentum
    elif setup_number == 3:
        df['CE_Signal'] = (df['STOCH_k'] > 60) & adx_cross_22_up & (df['Close'] > df['VWAP'])
        df['PE_Signal'] = (df['STOCH_k'] < 40) & adx_cross_22_up & (df['Close'] < df['VWAP'])

    # 4. EMA 9/21 Cross + volume expansion
    elif setup_number == 4:
        df['Prev_EMA_9'] = df['EMA_9'].shift(1)
        df['Prev_EMA_21'] = df['EMA_21'].shift(1)
        ema_cross_up = (df['Prev_EMA_9'] <= df['Prev_EMA_21']) & (df['EMA_9'] > df['EMA_21'])
        ema_cross_down = (df['Prev_EMA_9'] >= df['Prev_EMA_21']) & (df['EMA_9'] < df['EMA_21'])
        
        # Volume expansion: current volume > average of last 5
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        
        df['CE_Signal'] = ema_cross_up & vol_expansion
        df['PE_Signal'] = ema_cross_down & vol_expansion

    # 5. Opening Range Breakout (first 30 min)
    elif setup_number == 5:
        # Complex to do purely vectorized on a continuous stream without defining days,
        # but conceptually: break of the high/low of first 6 candles (30m on 5m chart)
        # Using a simplified daily max/min approach for the first 30 mins
        df['Time'] = df.index.time
        first_30m_mask = (df.index.hour == 9) & (df.index.minute <= 45) # 9:15 to 9:45
        
        df_daily = df.copy()
        df_daily['OR_High'] = df_daily['High'].where(first_30m_mask).groupby(df_daily.index.date).transform('max')
        df_daily['OR_Low'] = df_daily['Low'].where(first_30m_mask).groupby(df_daily.index.date).transform('min')
        
        # Breakout after 9:45
        after_30m_mask = ~first_30m_mask
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        
        df['CE_Signal'] = after_30m_mask & (df['Close'] > df_daily['OR_High']) & (df['Close'].shift(1) <= df_daily['OR_High']) & vol_expansion
        df['PE_Signal'] = after_30m_mask & (df['Close'] < df_daily['OR_Low']) & (df['Close'].shift(1) >= df_daily['OR_Low']) & vol_expansion

    # 6. Previous Day High/Low
    elif setup_number == 6:
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        df['CE_Signal'] = (df['Close'] > df['PDH']) & (df['Prev_High'] <= df['PDH']) & vol_expansion
        df['PE_Signal'] = (df['Close'] < df['PDL']) & (df['Prev_Low'] >= df['PDL']) & vol_expansion

    # 7. Inside Bar Breakout
    elif setup_number == 7:
        # Prev candle was inside bar, current candle breaks it
        df['Prev_Is_Inside'] = df['Is_Inside_Bar'].shift(1)
        ib_high = df['High'].shift(1)
        ib_low = df['Low'].shift(1)
        
        df['CE_Signal'] = df['Prev_Is_Inside'] & (df['Close'] > ib_high)
        df['PE_Signal'] = df['Prev_Is_Inside'] & (df['Close'] < ib_low)

    # 8. NR7 Breakout
    elif setup_number == 8:
        df['Prev_Is_NR7'] = df['Is_NR7'].shift(1)
        nr7_high = df['High'].shift(1)
        nr7_low = df['Low'].shift(1)
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        
        df['CE_Signal'] = df['Prev_Is_NR7'] & (df['Close'] > nr7_high) & vol_expansion
        df['PE_Signal'] = df['Prev_Is_NR7'] & (df['Close'] < nr7_low) & vol_expansion

    # 9. VWAP Reclaim/Reject
    elif setup_number == 9:
        df['Prev_Close'] = df['Close'].shift(1)
        # Reclaim: Was below VWAP, now crossed and closed above
        df['CE_Signal'] = (df['Prev_Close'] < df['VWAP'].shift(1)) & (df['Close'] > df['VWAP'])
        # Reject: Was above VWAP, now crossed and closed below
        df['PE_Signal'] = (df['Prev_Close'] > df['VWAP'].shift(1)) & (df['Close'] < df['VWAP'])

    # 10. Supertrend + Momentum
    elif setup_number == 10:
        df['Prev_ST_Dir'] = df['Supertrend_Dir'].shift(1)
        st_bullish_flip = (df['Prev_ST_Dir'] == -1) & (df['Supertrend_Dir'] == 1)
        st_bearish_flip = (df['Prev_ST_Dir'] == 1) & (df['Supertrend_Dir'] == -1)
        
        df['CE_Signal'] = st_bullish_flip & (df['STOCH_k'] > 60)
        df['PE_Signal'] = st_bearish_flip & (df['STOCH_k'] < 40)

    # 11. RSI Momentum
    elif setup_number == 11:
        rsi_cross_55_up = (df['Prev_RSI'] <= 55) & (df['RSI'] > 55)
        rsi_cross_45_down = (df['Prev_RSI'] >= 45) & (df['RSI'] < 45)
        
        df['CE_Signal'] = rsi_cross_55_up & (df['Close'] > df['VWAP'])
        df['PE_Signal'] = rsi_cross_45_down & (df['Close'] < df['VWAP'])

    # 12. Bollinger Expansion
    elif setup_number == 12:
        # Squeeze defined as Bandwidth < its 20-period moving average
        df['BB_Squeeze'] = df['BB_Bandwidth'] < df['BB_Bandwidth'].rolling(20).mean()
        df['Prev_Squeeze'] = df['BB_Squeeze'].shift(1)
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        
        # Breakout of squeeze
        df['CE_Signal'] = df['Prev_Squeeze'] & (df['Close'] > df['BB_Upper']) & vol_expansion
        df['PE_Signal'] = df['Prev_Squeeze'] & (df['Close'] < df['BB_Lower']) & vol_expansion

    # 13. Breakout + OI Confirmation
    elif setup_number == 13:
        # Price up, OI up, Volume up
        price_up = df['Close'] > df['Close'].shift(1)
        price_down = df['Close'] < df['Close'].shift(1)
        oi_up = df['OI'] > df['OI'].shift(1)
        vol_up = df['Volume'] > df['Volume'].shift(1)
        
        df['CE_Signal'] = price_up & oi_up & vol_up
        df['PE_Signal'] = price_down & oi_up & vol_up

    # 14. Liquidity Sweep Reversal
    elif setup_number == 14:
        # Sweeps recent 10-period low, but closes above
        df['Recent_Low'] = df['Low'].rolling(10).min().shift(1)
        df['Recent_High'] = df['High'].rolling(10).max().shift(1)
        
        sweep_low = (df['Low'] < df['Recent_Low']) & (df['Close'] > df['Recent_Low'])
        sweep_high = (df['High'] > df['Recent_High']) & (df['Close'] < df['Recent_High'])
        
        momentum_up = df['STOCH_k'] > df['STOCH_k'].shift(1)
        momentum_down = df['STOCH_k'] < df['STOCH_k'].shift(1)
        
        df['CE_Signal'] = sweep_low & momentum_up
        df['PE_Signal'] = sweep_high & momentum_down

    # 15. Multi-Confirmation Engine
    elif setup_number == 15:
        vol_expansion = df['Volume'] > df['Volume'].rolling(5).mean()
        
        ce_cond = (df['STOCH_k'] > 60) & (df['ADX'] > 22) & (df['Close'] > df['VWAP']) & (df['Close'] > df['VWMA']) & vol_expansion
        pe_cond = (df['STOCH_k'] < 40) & (df['ADX'] > 22) & (df['Close'] < df['VWAP']) & (df['Close'] < df['VWMA']) & vol_expansion
        
        # Require that it wasn't already in this state to generate a fresh signal
        df['Prev_CE_Cond'] = ce_cond.shift(1).fillna(False)
        df['Prev_PE_Cond'] = pe_cond.shift(1).fillna(False)
        
        df['CE_Signal'] = ce_cond & ~df['Prev_CE_Cond']
        df['PE_Signal'] = pe_cond & ~df['Prev_PE_Cond']

    return df

if __name__ == "__main__":
    from data_loader import fetch_data
    df = fetch_data()
    df = apply_indicators(df)
    
    print("Testing Strategy 1 (Stoch + VWAP Breakout)")
    df_sig1 = generate_signals(df, 1)
    ce_count = df_sig1['CE_Signal'].sum()
    pe_count = df_sig1['PE_Signal'].sum()
    print(f"CE Signals: {ce_count}, PE Signals: {pe_count}")
    
    print("Testing Strategy 15 (Multi-Confirmation)")
    df_sig15 = generate_signals(df, 15)
    ce_count15 = df_sig15['CE_Signal'].sum()
    pe_count15 = df_sig15['PE_Signal'].sum()
    print(f"CE Signals: {ce_count15}, PE Signals: {pe_count15}")
