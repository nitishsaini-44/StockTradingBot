import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

class PerformanceMetrics:
    """
    Calculate and track performance metrics for trading predictions
    """
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.confidence_scores = []
        
    def add_prediction(self, predicted_action, actual_action, confidence):
        """
        Add a single prediction with its actual outcome
        
        Args:
            predicted_action: Predicted action (0=HOLD, 1=BUY, 2=SELL)
            actual_action: Actual action that occurred (0=HOLD, 1=BUY, 2=SELL)
            confidence: Confidence score (0-100)
        """
        self.predictions.append(predicted_action)
        self.actuals.append(actual_action)
        self.confidence_scores.append(confidence)
    
    def get_accuracy(self):
        """Overall accuracy of predictions"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.actuals, self.predictions)
    
    def get_precision(self, action=None):
        """Precision for a specific action or weighted average"""
        if len(self.predictions) == 0:
            return 0.0
        
        if action is not None:
            # Binary precision for specific action
            binary_actuals = [1 if a == action else 0 for a in self.actuals]
            binary_preds = [1 if p == action else 0 for p in self.predictions]
            
            if sum(binary_preds) == 0:
                return 0.0
            return precision_score(binary_actuals, binary_preds, zero_division=0)
        else:
            # Weighted average precision
            return precision_score(self.actuals, self.predictions, average='weighted', zero_division=0)
    
    def get_recall(self, action=None):
        """Recall for a specific action or weighted average"""
        if len(self.predictions) == 0:
            return 0.0
        
        if action is not None:
            binary_actuals = [1 if a == action else 0 for a in self.actuals]
            binary_preds = [1 if p == action else 0 for p in self.predictions]
            
            if sum(binary_actuals) == 0:
                return 0.0
            return recall_score(binary_actuals, binary_preds, zero_division=0)
        else:
            return recall_score(self.actuals, self.predictions, average='weighted', zero_division=0)
    
    def get_f1_score(self, action=None):
        """F1 score for a specific action or weighted average"""
        if len(self.predictions) == 0:
            return 0.0
        
        if action is not None:
            binary_actuals = [1 if a == action else 0 for a in self.actuals]
            binary_preds = [1 if p == action else 0 for p in self.predictions]
            
            if sum(binary_actuals) == 0 and sum(binary_preds) == 0:
                return 0.0
            return f1_score(binary_actuals, binary_preds, zero_division=0)
        else:
            return f1_score(self.actuals, self.predictions, average='weighted', zero_division=0)
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        if len(self.predictions) == 0:
            return None
        return confusion_matrix(self.actuals, self.predictions, labels=[0, 1, 2])
    
    def get_action_distribution(self):
        """Get distribution of predicted and actual actions"""
        action_names = ['HOLD', 'BUY', 'SELL']
        pred_dist = {}
        actual_dist = {}
        
        for i, action in enumerate(action_names):
            pred_dist[action] = self.predictions.count(i)
            actual_dist[action] = self.actuals.count(i)
        
        return {'predicted': pred_dist, 'actual': actual_dist}
    
    def get_summary(self):
        """Get complete performance summary"""
        if len(self.predictions) == 0:
            return {}
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        summary = {
            'total_predictions': len(self.predictions),
            'accuracy': round(self.get_accuracy() * 100, 2),
            'precision_weighted': round(self.get_precision() * 100, 2),
            'recall_weighted': round(self.get_recall() * 100, 2),
            'f1_score_weighted': round(self.get_f1_score() * 100, 2),
            'action_performance': {},
            'average_confidence': round(np.mean(self.confidence_scores), 2) if self.confidence_scores else 0.0,
            'distribution': self.get_action_distribution()
        }
        
        # Per-action metrics
        for action_idx, action_name in action_names.items():
            summary['action_performance'][action_name] = {
                'precision': round(self.get_precision(action_idx) * 100, 2),
                'recall': round(self.get_recall(action_idx) * 100, 2),
                'f1_score': round(self.get_f1_score(action_idx) * 100, 2),
            }
        
        return summary
    
    def print_report(self):
        """Print a formatted performance report"""
        if len(self.predictions) == 0:
            print("No predictions to report")
            return
        
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE EVALUATION REPORT")
        print("="*60)
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"\n📈 Overall Metrics:")
        print(f"  Accuracy: {summary['accuracy']}%")
        print(f"  Precision (Weighted): {summary['precision_weighted']}%")
        print(f"  Recall (Weighted): {summary['recall_weighted']}%")
        print(f"  F1-Score (Weighted): {summary['f1_score_weighted']}%")
        print(f"  Average Confidence: {summary['average_confidence']}%")
        
        print(f"\n🎯 Per-Action Performance:")
        for action, metrics in summary['action_performance'].items():
            print(f"  {action}:")
            print(f"    Precision: {metrics['precision']}%")
            print(f"    Recall: {metrics['recall']}%")
            print(f"    F1-Score: {metrics['f1_score']}%")
        
        print(f"\n📊 Action Distribution:")
        dist = summary['distribution']
        print(f"  Predicted: {dist['predicted']}")
        print(f"  Actual: {dist['actual']}")
        
        print("\n" + "="*60 + "\n")


def evaluate_trading_strategy(portfolio_values, initial_balance):
    """
    Evaluate trading strategy performance metrics
    """
    returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        returns.append(ret)
    
    returns = np.array(returns)
    
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    # Calculate Sharpe Ratio (assuming 252 trading days and 2% risk-free rate)
    daily_return_mean = np.mean(returns)
    daily_return_std = np.std(returns)
    
    if daily_return_std > 0:
        sharpe_ratio = (daily_return_mean - 0.02/252) / daily_return_std * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Calculate Maximum Drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    # Win Rate (days with positive returns)
    win_days = np.sum(returns > 0)
    win_rate = (win_days / len(returns)) * 100 if len(returns) > 0 else 0
    
    metrics = {
        'total_return_pct': round(total_return, 2),
        'final_value': round(final_value, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'win_rate_pct': round(win_rate, 2),
        'avg_daily_return_pct': round(daily_return_mean * 100, 2)
    }
    
    return metrics
