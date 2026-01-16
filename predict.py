import pandas as pd
import torch
import numpy as np
from utils.features import get_state, extract_technical_indicators
from agents.dqn_agent import DQNAgent
from datetime import datetime, timedelta

# ===== CONFIG =====
CSV_PATH = "data/raw.csv"
MODEL_PATH = "model.pth"
SYMBOL = "TATSILV"

# Load latest data
df = pd.read_csv(CSV_PATH)

# Safety check
required_cols = ["Close", "SMA_5", "SMA_20", "Returns"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Load agent
agent = DQNAgent(state_size=4, action_size=3)
agent.load(MODEL_PATH)

# Get latest state
last_index = len(df) - 1
state = get_state(df, last_index)

# Predict
with torch.no_grad():
    q_values = agent.model(torch.tensor(state).unsqueeze(0))[0]
    probs = torch.softmax(q_values, dim=0)
    action = torch.argmax(probs).item()

mapping = {0: "HOLD", 1: "BUY", 2: "SELL"}
action_description = {
    0: "📌 HOLD - No significant movement expected",
    1: "📈 BUY - Price expected to increase",
    2: "📉 SELL - Price expected to decrease"
}

# Get technical indicators
indicators = extract_technical_indicators(df, last_index)
current_price = float(df.iloc[last_index]['Close'])
sma_5 = float(df.iloc[last_index]['SMA_5'])
sma_20 = float(df.iloc[last_index]['SMA_20'])

# Calculate next trading day
today = pd.to_datetime(df.iloc[last_index]['Date'])
next_day = today + timedelta(days=1)

# Determine trend direction
trend = "🟢 Bullish" if sma_5 > sma_20 else "🔴 Bearish"

# Output prediction
print("\n" + "="*60)
print("📊 NEXT DAY TRADING PREDICTION")
print("="*60)
print(f"📈 Symbol: {SYMBOL}")
print(f"📅 Today's Date: {df.iloc[last_index]['Date']}")
print(f"🗓️  Prediction For: {next_day.strftime('%Y-%m-%d')}")
print(f"\n💰 Current Price: ${current_price:.2f}")
print(f"📊 SMA(5): ${sma_5:.2f}")
print(f"📊 SMA(20): ${sma_20:.2f}")
print(f"📈 Trend: {trend}")
print(f"\n{'='*60}")
print(f"🤖 RECOMMENDATION: {mapping[action]}")
print(f"💡 {action_description[action]}")
print(f"🔥 Confidence: {float(probs[action]) * 100:.2f}%")
print(f"{'='*60}")

# Show all action probabilities
print(f"\n📊 Action Probabilities:")
for i, action_name in enumerate(['HOLD', 'BUY', 'SELL']):
    confidence = float(probs[i]) * 100
    bar_length = int(confidence / 2)
    bar = '█' * bar_length
    print(f"  {action_name}: {confidence:6.2f}% {bar}")

# Risk analysis
print(f"\n⚠️  Analysis Notes:")
if abs(indicators.get('price_change', 0)) > 2:
    print(f"  • High volatility detected: {indicators['price_change']:.2f}% daily change")
if sma_5 < sma_20:
    print(f"  • Downtrend: Short-term MA below long-term MA")
elif sma_5 > sma_20:
    print(f"  • Uptrend: Short-term MA above long-term MA")
else:
    print(f"  • Sideways market: MAs converging")

print(f"\n{'='*60}\n")

