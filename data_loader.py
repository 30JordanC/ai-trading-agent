import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", start="2018-01-01", end=None):
    data = yf.download(ticker, start=start, end=end)

    # Fix MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [col.lower() for col in data.columns]

    data = data[["open", "high", "low", "close", "volume"]]

    return data


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")

    print(df.head())

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/aapl_data.csv")

    print("Saved to data/aapl_data.csv")
