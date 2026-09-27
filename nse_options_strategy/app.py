import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data_loader import fetch_data
from indicators import apply_indicators
from strategies import generate_signals
from backtest import run_backtest

st.set_page_config(layout="wide", page_title="NSE Options Buying Setups")

st.title("🚀 NSE Options Buying Strategies Dashboard")
st.markdown("Test 15 different momentum & breakout setups designed for Options Buying (CE/PE).")

# Sidebar for configuration
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol (e.g. ^NSEI, ^NSEBANK, RELIANCE.NS)", value="^NSEI")
period = st.sidebar.selectbox("Lookback Period", ["5d", "1mo", "3mo", "1y"], index=0)

# Setup Dictionary
setups = {
    1: "Stoch + VWAP Breakout",
    2: "Stoch + VWMA + VWAP",
    3: "ADX Momentum",
    4: "EMA 9/21 Cross",
    5: "Opening Range Breakout",
    6: "Previous Day High/Low",
    7: "Inside Bar Breakout",
    8: "NR7 Breakout",
    9: "VWAP Reclaim/Reject",
    10: "Supertrend + Momentum",
    11: "RSI Momentum",
    12: "Bollinger Expansion",
    13: "Breakout + OI Confirmation",
    14: "Liquidity Sweep Reversal",
    15: "Multi-Confirmation Engine"
}

selected_setup_id = st.sidebar.selectbox(
    "Select Strategy Setup", 
    options=list(setups.keys()),
    format_func=lambda x: f"{x}. {setups[x]}"
)

# Load Data
@st.cache_data(ttl=300) # Cache for 5 mins
def load_and_process_data(sym, p, setup_id):
    df = fetch_data(symbol=sym, period=p)
    if not df.empty:
        df = apply_indicators(df)
        df = generate_signals(df, setup_number=setup_id)
    return df

with st.spinner("Fetching data and calculating indicators..."):
    df = load_and_process_data(symbol, period, selected_setup_id)

if df.empty:
    st.error("No data found for the given symbol and period.")
    st.stop()

# Backtest parameters
st.sidebar.subheader("Backtest Parameters")
sl_pct = st.sidebar.number_input("Stop Loss % (Underlying)", value=0.2, step=0.1) / 100.0
tp_pct = st.sidebar.number_input("Take Profit % (Underlying)", value=0.5, step=0.1) / 100.0

# Run Backtest
trades_df, summary = run_backtest(df, stop_loss_pct=sl_pct, take_profit_pct=tp_pct)

# Display Metrics
st.subheader(f"Strategy Performance: {setups[selected_setup_id]}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", summary['Total Trades'])
col2.metric("Win Rate", f"{summary['Win Rate (%)']}%")
col3.metric("Underlying Ret (sum)", f"{summary['Total Returns (Underlying % sum)']}%")
col4.metric("Max Drawdown", f"{summary['Max Drawdown (%)']}%")

# Chart
st.subheader("Price Chart & Signals")

# Create figure with secondary y-axis for Volume/Stochastics
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, subplot_titles=('Price & Signals', 'Volume'),
                    row_width=[0.2, 0.7])

# Candlestick
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price'), row=1, col=1)

# Add VWAP
if 'VWAP' in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='blue', width=1), name='VWAP'), row=1, col=1)

# Add Buy Signals (CE)
ce_signals = df[df['CE_Signal']]
if not ce_signals.empty:
    fig.add_trace(go.Scatter(x=ce_signals.index, y=ce_signals['Low'] - (df['Close'].mean()*0.001), 
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='CE Buy Signal'), row=1, col=1)

# Add Sell Signals (PE)
pe_signals = df[df['PE_Signal']]
if not pe_signals.empty:
    fig.add_trace(go.Scatter(x=pe_signals.index, y=pe_signals['High'] + (df['Close'].mean()*0.001), 
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='PE Buy Signal'), row=1, col=1)

# Volume bar chart
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='grey'), row=2, col=1)

fig.update_layout(height=600, xaxis_rangeslider_visible=False, template='plotly_dark',
                  margin=dict(l=0, r=0, t=30, b=0))

st.plotly_chart(fig, use_container_width=True)

# Trade History
st.subheader("Trade History")
if not trades_df.empty:
    st.dataframe(trades_df.style.format({
        'Entry_Price': '{:.2f}', 
        'Exit_Price': '{:.2f}',
        'PnL_Pct': '{:.2f}%'
    }), use_container_width=True)
else:
    st.info("No trades executed in this period.")

# Data Table
with st.expander("View Raw Data"):
    st.dataframe(df.tail(100))
