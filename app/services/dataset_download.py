import yfinance as yf
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
print(data.head())
data.to_csv("apple_stock_data.csv")

print("✅ Data saved successfully as apple_stock_data.csv")