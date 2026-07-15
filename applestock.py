import yfinance as yf 
import matplotlib.pyplot as plt
data = yf.download("AAPL", period="1y")
data["Close"].plot(title="Apple — Last 1 Year")
print(data.head(5))
plt.show()
