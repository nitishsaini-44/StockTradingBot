import streamlit as st
import pandas as pd
import torch
from datetime import timedelta
import os
import subprocess

from envs.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from utils.features import get_state, extract_technical_indicators, calculate_next_day_label
from utils.metrics import PerformanceMetrics

st.set_page_config(page_title="Stock DQN Trading Bot", layout="wide")

# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_data(csv_path="data/raw.csv"):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_agent():
    try:
        agent = DQNAgent(state_size=4, action_size=3)
        agent.load("model.pth")
        return agent
    except:
        return None

def download_company_data(symbol):
    """Download data for a specific company symbol"""
    try:
        with st.spinner(f"Downloading data for {symbol}..."):
            # Use the virtual environment's Python executable
            python_exe = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                # Fallback to 'python' command
                python_exe = "python"
            
            # Run the download_data.py script with the symbol
            result = subprocess.run(
                [python_exe, "download_data.py", symbol],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                st.success(f"✅ Data downloaded successfully for {symbol}!")
                # Clear the cache to reload new data
                st.cache_data.clear()
                return True
            else:
                st.error(f"❌ Error downloading data: {result.stderr}")
                return False
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# Initialize session state for symbol
last_symbol_path = "data/last_symbol.txt"
default_symbol = "IDEA"
if os.path.exists(last_symbol_path):
    try:
        with open(last_symbol_path, "r") as f:
            default_symbol = f.read().strip()
    except:
        pass

if "symbol" not in st.session_state:
    st.session_state.symbol = default_symbol

data = load_data()
agent = load_agent()

# -------------------------
# Title
# -------------------------
st.title("📈 Stock DQN Trading Bot")
st.caption("AI-Powered Trading Decision Support System")

# -------------------------
# Company Selection & Data Download
# -------------------------
st.subheader("🔍 Company Data Management")

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("📍 Enter Company Symbol", value=st.session_state.symbol, placeholder="e.g., IDEA, TCS, INFY")
    st.session_state.symbol = symbol.upper()

with col2:
    st.write("")  # Add spacing
    if st.button("⬇️ Download Data", key="download_btn"):
        if symbol.strip():
            if download_company_data(symbol.upper()):
                # Reload data after successful download
                st.rerun()
        else:
            st.error("Please enter a symbol")

with col3:
    st.write("")  # Add spacing
    if st.button("🔄 Refresh Data", key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

# -------------------------
# System Status
# -------------------------
st.subheader("🔧 System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Data Loaded", "Yes" if data is not None else "No")

with col2:
    st.metric("Model Loaded", "Yes" if agent is not None else "No")

with col3:
    st.metric("Data Points", len(data) if data is not None else 0)

with col4:
    if data is not None:
        st.metric("Date Range", f"{data.iloc[-1]['Date']}")
    else:
        st.metric("Date Range", "N/A")

# -------------------------
# Data Summary
# -------------------------
st.subheader("📊 Data Summary")

if data is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Latest Close", f"{data.iloc[-1]['Close']:.2f}")
        st.metric("SMA 5", f"{data.iloc[-1]['SMA_5']:.2f}")
        st.metric("SMA 20", f"{data.iloc[-1]['SMA_20']:.2f}")

    with col2:
        st.metric("Min Price", f"{data['Close'].min():.2f}")
        st.metric("Max Price", f"{data['Close'].max():.2f}")
        st.metric("Mean Price", f"{data['Close'].mean():.2f}")

    with col3:
        st.metric("Total Volume", int(data['Volume'].sum()))
        st.metric("Avg Volume", int(data['Volume'].mean()))

# -------------------------
# Chart
# -------------------------
st.subheader("📈 Last 30 Days Price Chart")

if data is not None:
    chart_df = data.tail(30)
    chart_df = chart_df.set_index("Date")

    st.line_chart(chart_df[["Close", "SMA_5", "SMA_20"]])

# -------------------------
# Training
# -------------------------
st.subheader("⚙️ Train Model")

episodes = st.number_input("Episodes", min_value=10, max_value=1000, value=300)

if st.button("Train"):
    if data is None:
        st.error("Data not loaded.")
    else:
        with st.spinner("Training model..."):
            agent_new = DQNAgent(state_size=4, action_size=3)
            env_train = TradingEnvironment(data)

            for e in range(episodes):
                state = env_train.reset()
                done = False

                while not done:
                    action = agent_new.act(state)
                    next_state, reward, done, info = env_train.step(action)
                    agent_new.remember(state, action, reward, next_state, done)
                    state = next_state

                agent_new.replay()

            agent_new.save()

        st.success("Model trained and saved successfully!")

# -------------------------
# Accuracy
# -------------------------
st.subheader("📊 Model Accuracy")

if st.button("Calculate Accuracy"):
    if data is None or agent is None:
        st.error("Data or model not loaded.")
    else:
        metrics = PerformanceMetrics()

        test_range = min(50, len(data) - 10)
        start_idx = len(data) - test_range

        for idx in range(start_idx, len(data) - 1):
            try:
                state = get_state(data, idx)
                with torch.no_grad():
                    q_values = agent.model(torch.tensor(state).unsqueeze(0))[0]
                    probs = torch.softmax(q_values, dim=0)
                    predicted_action = torch.argmax(probs).item()
                    confidence = float(probs[predicted_action]) * 100

                actual_action = calculate_next_day_label(data, idx)

                if actual_action is not None:
                    metrics.add_prediction(predicted_action, actual_action, confidence)
            except:
                continue

        summary = metrics.get_summary()

        if summary:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{summary['accuracy']:.2f}%")
            col2.metric("Precision", f"{summary['precision_weighted']:.2f}%")
            col3.metric("Recall", f"{summary['recall_weighted']:.2f}%")
            col4.metric("F1 Score", f"{summary['f1_score_weighted']:.2f}%")

# -------------------------
# Prediction
# -------------------------
st.subheader("🤖 Next Day Prediction")

if st.button("Get Prediction"):
    if data is None or agent is None:
        st.error("Data or model not loaded.")
    else:
        last_index = len(data) - 1
        state = get_state(data, last_index)

        with torch.no_grad():
            q_values = agent.model(torch.tensor(state).unsqueeze(0))[0]
            probs = torch.softmax(q_values, dim=0)
            action = torch.argmax(probs).item()

        indicators = extract_technical_indicators(data, last_index)

        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_desc = {
            0: 'No significant movement expected',
            1: 'Price expected to increase',
            2: 'Price expected to decrease'
        }

        today = pd.to_datetime(data.iloc[last_index]['Date'])
        next_day = today + timedelta(days=1)

        st.success(f"Recommendation: **{action_map[action]}**")
        st.write(action_desc[action])
        st.write(f"Confidence: **{float(probs[action]) * 100:.2f}%**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HOLD", f"{float(probs[0])*100:.1f}%")
        with col2:
            st.metric("BUY", f"{float(probs[1])*100:.1f}%")
        with col3:
            st.metric("SELL", f"{float(probs[2])*100:.1f}%")

        st.write("### Market Info")
        st.write(f"Current Price: {data.iloc[last_index]['Close']:.2f}")
        st.write(f"SMA 5: {data.iloc[last_index]['SMA_5']:.2f}")
        st.write(f"SMA 20: {data.iloc[last_index]['SMA_20']:.2f}")
        st.write(f"Prediction Date: {next_day.date()}")

