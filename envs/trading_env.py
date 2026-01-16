import numpy as np
from utils.features import get_state
from utils.position_sizing import calculate_position

class TradingEnvironment:
    """
    Trading Environment for DQN Agent Training
    
    Simulates a trading scenario where the agent can:
    - HOLD (0): Do nothing
    - BUY (1): Buy 20% of available capital in shares
    - SELL (2): Sell all held shares
    
    The environment provides:
    - State: [Close, SMA_5, SMA_20, Returns]
    - Reward: Change in portfolio value
    - Done: Episode termination flag
    """
    
    def __init__(self, data, position_pct=0.2, initial_balance=458.19):
        """
        Initialize Trading Environment
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            position_pct: Percentage of balance to risk per trade (default 20%)
            initial_balance: Starting capital (default $458.19)
        """
        self.data = data
        self.position_pct = position_pct
        self.initial_balance = initial_balance
        
        # Tracking variables for training
        self.balance = initial_balance
        self.shares = 0
        self.index = 0
        self.prev_value = initial_balance
        
        # Episode statistics (for monitoring training progress)
        self.total_profit = 0
        self.trades_executed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_portfolio_value = initial_balance
        self.min_portfolio_value = initial_balance
        self.episode_rewards = []

    def reset(self):
        """
        Reset environment for new training episode
        
        Returns:
            Initial state vector [Close, SMA_5, SMA_20, Returns]
        """
        self.balance = self.initial_balance
        self.shares = 0
        self.index = 0
        self.prev_value = self.initial_balance
        
        # Reset episode statistics
        self.total_profit = 0
        self.trades_executed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_portfolio_value = self.initial_balance
        self.min_portfolio_value = self.initial_balance
        self.episode_rewards = []
        
        return get_state(self.data, self.index)

    def step(self, action):
        """
        Execute one step of trading (one day/candle)
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            next_state: State vector for next day
            reward: Change in portfolio value (profit/loss signal)
            done: True if episode is finished
            info: Dictionary with additional info (portfolio_value, shares, balance)
        """
        # Get current price from data
        current_price = float(self.data.loc[self.index, 'Close'])
        
        # Track previous portfolio value for reward calculation
        previous_portfolio_value = self.balance + (self.shares * current_price)
        
        # Execute trading action
        # ===== BUY ACTION =====
        if action == 1 and self.shares == 0:  # Only buy if not already holding
            shares_to_buy = calculate_position(self.balance, current_price, self.position_pct)
            cost = shares_to_buy * current_price
            
            # Execute only if we have enough balance
            if cost <= self.balance and shares_to_buy > 0:
                self.balance -= cost
                self.shares += shares_to_buy
                self.trades_executed += 1
        
        # ===== SELL ACTION =====
        elif action == 2 and self.shares > 0:  # Only sell if holding shares
            sell_proceeds = self.shares * current_price
            self.balance += sell_proceeds
            self.shares = 0
            self.trades_executed += 1
        
        # ===== HOLD ACTION (action == 0) =====
        # No action needed
        
        # Move to next time step
        self.index += 1
        
        # Check if episode is done (reached end of data)
        done = self.index >= len(self.data) - 1
        
        # Calculate new portfolio value
        next_price = float(self.data.loc[self.index, 'Close']) if not done else current_price
        portfolio_value = self.balance + (self.shares * next_price)
        
        # Calculate reward (key signal for agent training!)
        # Positive reward for profit, negative for loss
        reward = portfolio_value - self.prev_value
        self.prev_value = portfolio_value
        self.episode_rewards.append(reward)
        
        # Track winning/losing trades
        if reward > 0.01:  # Small threshold to avoid floating point noise
            self.winning_trades += 1
        elif reward < -0.01:
            self.losing_trades += 1
        
        # Track portfolio extremes
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        self.min_portfolio_value = min(self.min_portfolio_value, portfolio_value)
        self.total_profit = portfolio_value - self.initial_balance
        
        # Get next state (features for agent to observe)
        next_state = get_state(self.data, self.index) if not done else None
        
        # Return environment step information
        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "shares": self.shares,
            "current_price": current_price,
            "total_profit": self.total_profit,
            "trades": self.trades_executed,
            "max_value": self.max_portfolio_value,
            "min_value": self.min_portfolio_value,
        }
        
        return next_state, reward, done, info
    
    def get_episode_stats(self):
        """
        Get statistics from completed episode (for monitoring training)
        
        Returns:
            Dictionary with episode performance metrics
        """
        total_reward = sum(self.episode_rewards) if self.episode_rewards else 0
        win_rate = (self.winning_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0
        
        return {
            "total_profit": self.total_profit,
            "total_reward": total_reward,
            "trades": self.trades_executed,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "max_portfolio_value": self.max_portfolio_value,
            "min_portfolio_value": self.min_portfolio_value,
            "final_portfolio_value": self.balance + (self.shares * float(self.data.loc[self.index - 1, 'Close'])),
        }
    
    def get_current_portfolio_value(self):
        """Get current portfolio value (balance + stock value)"""
        current_price = float(self.data.loc[self.index, 'Close']) if self.index < len(self.data) else 0
        return self.balance + (self.shares * current_price)
