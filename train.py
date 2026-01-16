import pandas as pd
from envs.trading_env import TradingEnvironment
from agents.dqn_agent import DQNAgent
from utils.features import calculate_next_day_label
from utils.metrics import PerformanceMetrics, evaluate_trading_strategy

data = pd.read_csv("data/raw.csv")

env = TradingEnvironment(data)
agent = DQNAgent(state_size=4, action_size=3)
metrics = PerformanceMetrics()

episodes = 300

print("\n" + "="*60)
print("🚀 STARTING DQN MODEL TRAINING")
print("="*60)
print(f"Episodes: {episodes}")
print(f"Data points: {len(data)}")
print("="*60 + "\n")

for e in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    episode_predictions = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        # Get actual label for this step (next day's actual action)
        actual_label = calculate_next_day_label(data, env.index - 1)
        if actual_label is not None:
            # Get confidence from Q-values while state is valid
            import torch
            with torch.no_grad():
                q_values = agent.model(torch.tensor(state).unsqueeze(0))[0]
                confidence = torch.max(torch.softmax(q_values, dim=0)).item() * 100
            episode_predictions.append((action, actual_label, confidence))
        
        state = next_state
        total_reward += reward
    
    agent.replay()
    
    # Add episode predictions to metrics
    for pred, actual, confidence in episode_predictions:
        metrics.add_prediction(pred, actual, confidence)

    if (e + 1) % 50 == 0:
        print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}")

agent.save()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print(f"Model saved to: model.pth\n")

# Print metrics
metrics.print_report()