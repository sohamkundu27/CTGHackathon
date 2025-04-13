import pandas as pd
import json
import numpy as np

# Read the CSV file into a DataFrame
model_state = pd.read_csv('model_state.csv')

# Rename columns to more expected names
model_state = model_state.rename(columns={
    'adjusted_open_1d': 'open',
    'adjusted_high_1d': 'high',
    'adjusted_low_1d': 'low',
    'adjusted_close_1d': 'close',
    'volume_1d': 'volume'
})

# Load asset information from a JSON file
asset_info = json.load(open('asset_info.json'))

# Function to get all equity tickers using asset_info.json
def get_equity_tickers(assets):
    equity_tickers = []
    for asset in assets:
        if asset["UnderlyingAssetClass"] == "Equity":
            equity_tickers.append(asset["Ticker"])
    return equity_tickers

# Get the equity tickers from asset_info
equity_tickers = get_equity_tickers(asset_info)

# Filter for equity tickers in your dataset
model_state = model_state[model_state['ticker'].isin(equity_tickers)]

# Example: Apply strategy to one ticker at a time (here we take the first ticker)
ticker = equity_tickers[0]
df = model_state[model_state['ticker'] == ticker].copy()
df.sort_values('date', inplace=True)

# Calculate moving averages for a pure moving average crossover strategy
df['ma_short'] = df['close'].rolling(window=10).mean()
df['ma_long'] = df['close'].rolling(window=30).mean()

# Generate signals based solely on moving average crossovers:
# Signal = 1 when short MA is above long MA, and -1 otherwise.
df['signal'] = 0
df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
df.loc[df['ma_short'] < df['ma_long'], 'signal'] = -1

# Forward-fill signals to simulate holding positions
df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

# Calculate strategy returns using log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['strategy_return'] = df['position'].shift(1) * df['log_return']
df['cumulative_return'] = df['strategy_return'].cumsum()

# Print the relevant columns of the last few rows
print(df[['date', 'close', 'ma_short', 'ma_long', 'signal', 'position', 'cumulative_return']].tail())
