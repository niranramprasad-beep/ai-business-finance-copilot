# Day 2 mini-challenge: Nike's highest closing price this year
# Run with: python3 nike_challenge.py
import yfinance as yf

# Download Nike stock data for this year, flat columns
nike = yf.download("NKE", start="2026-01-01", multi_level_index=False)

# Find the highest closing price and the date it happened
highest_close = float(nike["Close"].max())
highest_date = nike["Close"].idxmax().date()

print(f"Nike's highest closing price this year: ${highest_close:.2f}")
print(f"It happened on: {highest_date}")