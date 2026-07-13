# Day 1 win: download Apple's stock history and plot the last year
# Run with: python3 apple_stock.py
# (Needs: pip3 install yfinance plotly)

import yfinance as yf
import plotly.express as px

# 1. Ask Yahoo's API (through the yfinance library) for 1 year of Apple data
data = yf.download("AAPL", period="1y")

# 2. Peek at what came back — a table (DataFrame) of dates and prices
print(data.head())        # first 5 rows
print(data.shape)         # (rows, columns)

# 3. Plot the closing price over the last year
fig = px.line(data, y=data["Close"].squeeze(), title="Apple (AAPL) — Last 1 Year")
fig.show()  # opens the chart in your browser
