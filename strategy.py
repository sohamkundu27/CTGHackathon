import pandas as pd
import json
import numpy as np


model_state = pd.read_csv('model_state.csv')
# Rename columns to be more expected

model_state = model_state.rename(columns={
    'adjusted_open_1d': 'open',
    'adjusted_high_1d': 'high',
    'adjusted_low_1d': 'low',
    'adjusted_close_1d': 'close',
    'volume_1d': 'volume'
})

asset_info = json.load(open('asset_info.json'))

# Function to get all equity tickers using asset_info.json
def get_equity_tickers(assets):
    equity_tickers = []
    for asset in assets:
        if asset["UnderlyingAssetClass"] == "Equity":
            equity_tickers.append(asset["Ticker"])
    return equity_tickers

# Get and print the equity tickers
equity_tickers = get_equity_tickers(asset_info)

# Filter for equity tickers in your dataset
model_state = model_state[model_state['ticker'].isin(equity_tickers)]

# Example: Apply strategy to one ticker at a time (you can loop through equity_tickers)
ticker = equity_tickers[0]
df = model_state[model_state['ticker'] == ticker].copy()
df.sort_values('date', inplace=True)

# Calculate moving averages
df['ma_short'] = df['close'].rolling(window=10).mean()
df['ma_long'] = df['close'].rolling(window=30).mean()

# Calculate rolling volatility (20-day standard deviation of log returns)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['log_return'].rolling(window=20).std()

# Define volatility threshold
vol_threshold = df['volatility'].quantile(0.7)

# Generate signals
df['signal'] = 0
df.loc[(df['ma_short'] > df['ma_long']) & (df['volatility'] < vol_threshold), 'signal'] = 1
df.loc[(df['ma_short'] < df['ma_long']) | (df['volatility'] > vol_threshold), 'signal'] = -1

# Forward-fill signals to simulate holding positions
df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

# Optional: calculate strategy returns
df['strategy_return'] = df['position'].shift(1) * df['log_return']
df['cumulative_return'] = df['strategy_return'].cumsum()

print(df[['date', 'close', 'ma_short', 'ma_long', 'volatility', 'signal', 'position', 'cumulative_return']].tail())    