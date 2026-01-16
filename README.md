# Stock DQN Trading Bot

A machine learning-based trading bot that uses Deep Q-Learning (DQN) to predict and make trading decisions on stock market data.

## 📋 Overview

This project implements a reinforcement learning agent using Deep Q-Networks to analyze stock price movements and make buy/hold/sell decisions. The bot can be trained on historical stock data and deployed for real-time predictions through a Streamlit web interface.

## 🎯 Features

- **Deep Q-Learning Agent**: Advanced reinforcement learning model for trading decisions
- **Technical Indicators**: Implements SMA (Simple Moving Average) and Returns calculations
- **Trading Environment**: Custom OpenAI Gym-like environment for simulating stock trading
- **Web Interface**: Streamlit-based dashboard for visualizing predictions and performance
- **Real-time Data**: Downloads live stock data using Yahoo Finance
- **Performance Metrics**: Tracks accuracy, precision, recall, and other trading metrics
- **Position Sizing**: Intelligent position sizing strategies

## 📁 Project Structure

```
stock_dqn_bot/
├── stream.py                  # Streamlit web application
├── train.py                   # Model training script
├── predict.py                 # Prediction script
├── download_data.py           # Data fetching script
├── model.pth                  # Trained model weights
├── requirements.txt           # Python dependencies
├── agents/
│   └── dqn_agent.py          # DQN agent implementation
├── envs/
│   └── trading_env.py         # Custom trading environment
├── models/
│   └── dqn.py                # DQN neural network architecture
├── utils/
│   ├── features.py           # Feature extraction & state generation
│   ├── metrics.py            # Performance metrics calculation
│   └── position_sizing.py    # Position sizing strategies
└── data/
    ├── raw.csv               # Historical stock data
    └── last_symbol.txt       # Last selected stock symbol
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone or navigate to the project directory:
```bash
cd stock_dqn_bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Download Stock Data

Download historical data for a specific stock symbol:

```bash
python download_data.py
```

### Train the Model

Train the DQN agent on the downloaded data:

```bash
python train.py
```

This will:
- Load the historical data from `data/raw.csv`
- Train the agent for 300 episodes
- Save the trained model to `model.pth`
- Display training metrics

### Make Predictions

Make predictions on the latest data point:

```bash
python predict.py
```

### Web Interface

Run the interactive Streamlit dashboard:

```bash
streamlit run stream.py
```

This provides:
- Real-time stock price visualization
- Live trading predictions
- Model performance metrics
- Historical prediction accuracy
- Data download interface

## 🧠 Model Architecture

### DQN Agent

The agent consists of:
- **State Size**: 4 features (technical indicators)
- **Action Size**: 3 actions (Buy=0, Hold=1, Sell=2)
- **Neural Network**: Deep Q-Network with replay memory
- **Learning**: Q-learning with experience replay and target network

### Features Used

- Close Price
- Simple Moving Average (SMA-5)
- Simple Moving Average (SMA-20)

### Trading Environment

Custom environment that:
- Simulates trading based on agent actions
- Calculates rewards based on returns
- Provides state observations

## 📊 Performance Metrics

The system tracks:
- **Accuracy**: Correct prediction percentage
- **Precision**: True positives vs all positives
- **Recall**: True positives vs all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Cumulative Returns**: Total profit/loss from trades
- **Sharpe Ratio**: Risk-adjusted returns

## 🔧 Configuration

Key parameters in scripts:

- `episodes`: Number of training episodes (default: 300)
- `state_size`: Number of features (default: 4)
- `action_size`: Number of actions (default: 3)
- `SYMBOL`: Stock ticker symbol (e.g., "AAPL", "TATSILV")
- Learning rate, discount factor, and other hyperparameters in agent class

## 📦 Dependencies

- **torch**: Deep learning framework for neural networks
- **pandas**: Data manipulation and analysis
- **yfinance**: Yahoo Finance data downloader
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web interface framework
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **flask**: API server (optional)

See `requirements.txt` for specific versions.

## 🎓 How It Works

1. **Data Collection**: Download historical OHLCV data for a stock
2. **Feature Engineering**: Calculate technical indicators
3. **Training**: Agent learns trading patterns through reinforcement learning
   - Takes actions based on state observations
   - Receives rewards based on trading performance
   - Stores experiences in replay memory
   - Updates neural network weights
4. **Prediction**: Model makes buy/hold/sell decisions on new data
5. **Evaluation**: Performance metrics track accuracy and returns

## 📝 Notes

- Model performance depends on data quality and quantity
- Past performance does not guarantee future results
- This is for educational purposes; use with caution in live trading
- Always backtest thoroughly before deploying to real accounts

## 🤝 Contributing

Feel free to modify and improve the project. Areas for enhancement:
- Additional technical indicators
- Different neural network architectures
- Multiple timeframe analysis
- Portfolio optimization
- Risk management strategies

## 📄 License

This project is open-source and available for personal and educational use.

## ⚠️ Disclaimer

This project is for educational purposes only. Stock trading involves significant risk. Do not use this bot for actual trading without thorough testing and understanding of the risks involved. The creator is not responsible for any financial losses.
