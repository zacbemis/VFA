# VFA - Value & Forecast Analytics

A comprehensive stock market analysis toolkit combining machine learning prediction models with fundamental valuation analysis. This project demonstrates proficiency in financial data analysis, deep learning, and quantitative trading strategies.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/ea82eccc-fa95-4c2a-8922-ffbefd55a542" />


## 🎯 Project Overview

VFA integrates multiple approaches to stock market analysis:
- **Deep Learning Price Prediction** using LSTM neural networks
- **Classification-Based Trading Signals** with Random Forest algorithms
- **Fundamental Valuation Analysis** using financial ratios (P/E, P/B)

## 🔧 Technical Stack

- **Python 3.x**
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **Data Processing**: NumPy, Pandas
- **Financial Data**: yfinance API
- **Visualization**: Matplotlib

## 📊 Features

### 1. LSTM Stock Price Prediction
**File**: `LSTM stock predict.py`

- Implements a sequential LSTM neural network for time-series forecasting
- Uses 60-day lookback window for pattern recognition
- Trained on 13+ years of historical stock data (2010-2023)
- Visualizes actual vs. predicted stock prices with interactive plots

**Key Technical Highlights**:
- Data normalization using MinMaxScaler
- Two-layer LSTM architecture (50 units each)
- 80/20 train-test split for model validation
- Mean squared error optimization

### 2. Random Forest Trading Strategy
**File**: `RandomForestClass stock predict.py`

- Binary classification model predicting next-day price movements
- Advanced feature engineering with multiple time horizons (2, 5, 60, 250, 1000 days)
- Walk-forward backtesting methodology for realistic performance evaluation
- Precision-optimized predictions with 60% confidence threshold

**Key Technical Highlights**:
- Rolling average ratios and trend indicators
- 200-tree Random Forest ensemble
- Time-series cross-validation with 250-day test windows
- Visual buy signal identification on price charts

### 3. Fundamental Value Calculator
**File**: `ValueCalculator.py`

- Real-time stock valuation using fundamental metrics
- P/E (Price-to-Earnings) and P/B (Price-to-Book) ratio analysis
- Comparative valuation against sector benchmarks
- Interactive CLI for on-demand stock analysis

## 🚀 Usage

### LSTM Price Prediction
```python
python "LSTM stock predict.py"
```
Generates price predictions and comparison plots for AAPL stock.

### Random Forest Trading Signals
```python
python "RandomForestClass stock predict.py"
```
Outputs precision score and visualizes buy signals on historical data.

### Value Analysis
```python
python ValueCalculator.py
```
Enter any stock ticker to receive fundamental valuation analysis.

## 📈 Results & Performance

- **LSTM Model**: Captures long-term price trends with visual accuracy assessment
- **Random Forest**: Achieves measurable precision in directional predictions with backtested results
- **Value Calculator**: Provides instant fundamental analysis for investment decisions

## 🎓 Skills Demonstrated

- **Machine Learning**: Deep learning (LSTM), ensemble methods (Random Forest)
- **Financial Analysis**: Technical indicators, fundamental ratios, backtesting
- **Data Engineering**: Time-series preprocessing, feature engineering, data normalization
- **Software Development**: Modular code design, API integration, data visualization

## 📦 Installation

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
```

## 🔮 Future Enhancements

- Multi-stock portfolio optimization
- Real-time prediction API
- Sentiment analysis integration from news/social media
- Interactive web dashboard with React/Flask
- Extended fundamental metrics (DCF, dividend models)

## 📝 License

This project is available for educational and portfolio purposes.

---

**Note**: This project is for educational purposes only. Stock predictions should not be used as the sole basis for investment decisions.
