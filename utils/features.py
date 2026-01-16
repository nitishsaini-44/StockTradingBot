import numpy as np

def get_state(data, index):
    return np.array([
        data.loc[index, 'Close'],
        data.loc[index, 'SMA_5'],
        data.loc[index, 'SMA_20'],
        data.loc[index, 'Returns']
    ], dtype=np.float32)

def calculate_next_day_label(data, index):
    """
    Calculate the actual label for next day (Buy=1, Sell=2, Hold=0)
    Buy: Next day's close > today's close (price going up)
    Sell: Next day's close < today's close (price going down)
    Hold: Next day's close == today's close (no significant change)
    """
    if index >= len(data) - 1:
        return None
    
    current_close = data.loc[index, 'Close']
    next_close = data.loc[index + 1, 'Close']
    
    threshold = 0.01  # 1% threshold
    price_change = (next_close - current_close) / current_close
    
    if price_change > threshold:
        return 1  # BUY
    elif price_change < -threshold:
        return 2  # SELL
    else:
        return 0  # HOLD

def extract_technical_indicators(data, index):
    """
    Extract technical indicators for analysis
    """
    if index < 1:
        return {}
    
    current_close = data.loc[index, 'Close']
    prev_close = data.loc[index - 1, 'Close']
    
    indicators = {
        'close': current_close,
        'sma_5': data.loc[index, 'SMA_5'],
        'sma_20': data.loc[index, 'SMA_20'],
        'returns': data.loc[index, 'Returns'],
        'price_change': ((current_close - prev_close) / prev_close) * 100,
    }
    
    # Add momentum indicators
    if 'SMA_5' in data.columns and 'SMA_20' in data.columns:
        indicators['sma_crossover'] = data.loc[index, 'SMA_5'] > data.loc[index, 'SMA_20']
    
    return indicators
