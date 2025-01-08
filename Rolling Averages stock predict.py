import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import yfinance as yf
import matplotlib.pyplot as plt

# Load and process data
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Tomorrow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
    return data

# Feature engineering
def feature_engineering(data):
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = data['Close'].rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data['Close'] / rolling_averages
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data['Target'].shift(1).rolling(horizon).sum()
        new_predictors += [ratio_column, trend_column]

    return data.dropna(), new_predictors

# Backtesting with Random Forest model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

if __name__ == "__main__":
    # Fetch data for a specific stock (e.g., Apple)
    ticker = "AAPL"
    start_date = "1990-01-01"
    end_date = "2024-01-01"
    data = fetch_stock_data(ticker, start=start_date, end=end_date)

    # Engineer features
    data, new_predictors = feature_engineering(data)

    # Backtest the strategy
    predictions = backtest(data, model, new_predictors)

    # Calculate precision score
    precision = precision_score(predictions['Target'], predictions['Predictions'])
    print(f"Precision Score: {precision:3f}")

    # Plotting the graph
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Actual Close Prices', color='blue', alpha=0.6)

    # Highlight buy signals (predicted upward movements)
    buy_signals = predictions[predictions['Predictions'] == 1]
    plt.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'],
                color='green', label='Predicted Buy Signal', marker='^', alpha=0.8)
    plt.legend()
    plt.title("Stock Price and Predicted Buy Signals")
    plt.show()


