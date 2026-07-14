import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ---------- Page setup ----------
st.set_page_config(page_title="Mini Finance Copilot", layout="wide")
st.title("Mini Finance Copilot")
st.caption("Educational analytics — not investment advice")

# ---------- Sidebar (user inputs = the arguments!) ----------
st.sidebar.header("Controls")
ticker = st.sidebar.selectbox(
    "Pick a company",
    ["AAPL", "NKE", "TSLA", "DIS", "MCD", "MSFT", "GOOGL", "AMZN"]
)
period = st.sidebar.selectbox(
    "Time period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=3
)

# ---------- Function that does the work ----------
def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    # yfinance sometimes returns multi-level columns; flatten them
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    return data

data = get_stock_data(ticker, period)

if data.empty:
    st.error("No data came back. Check your internet or try another ticker.")
    st.stop()

# ---------- Math ----------
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

current_price = float(data["Close"].iloc[-1])
start_price = float(data["Close"].iloc[0])
pct_change = (current_price - start_price) / start_price * 100
period_high = float(data["Close"].max())
period_low = float(data["Close"].min())

# ---------- KPI cards ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric(f"Change over {period}", f"{pct_change:+.2f}%")
col3.metric("Period High", f"${period_high:,.2f}")
col4.metric("Period Low", f"${period_low:,.2f}")

# ---------- Chart ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"],
                         name="Close Price", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["MA20"],
                         name="20-day avg", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"],
                         name="50-day avg", line=dict(dash="dash")))
fig.update_layout(title=f"{ticker} — Price with Moving Averages",
                  xaxis_title="Date", yaxis_title="Price ($)",
                  height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------- Plain-English insight ----------
st.subheader("Quick take")
direction = "up" if pct_change > 0 else "down"
st.write(
    f"{ticker} is {direction} {abs(pct_change):.1f}% over the last {period}. "
    f"It traded between ${period_low:,.2f} and ${period_high:,.2f} in that window."
)

with st.expander("What am I looking at?"):
    st.write(
        "- Close Price: what the stock ended each day at.\n"
        "- 20-day avg: the average of the last 20 days — a smoothed short-term trend line.\n"
        "- 50-day avg: same idea but slower — the longer-term trend.\n"
        "- When the price is above both lines, the stock has been trending up recently."
    )
