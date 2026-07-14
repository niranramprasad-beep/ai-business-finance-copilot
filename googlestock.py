import yfinance as yf
import pandas as pd
import plotly.express as px

data = yf.download("GOOG", period="1y") #Google stock data for 1 year
print(data.head(15))
print(data.shape)

fig = px.line(data, y=data["Close"].squeeze(), title="Google (GOOG) — Last 1 Year")
fig.show()  # opens the chart in your browser