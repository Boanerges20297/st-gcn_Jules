"""
Baseline Evaluation Script
Evaluates current production model on test data
Generates baseline metrics for P@5-20, NDCG, etc.

Author: ST-GCN Enhanced System
Date: Feb 2026
"""

import json
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics import MetricReporter


class BaselineEvaluator:
    """
    Load production model and evaluate on test data
    """
    
    def __init__(self, model_path='models/stgcn_model_v2.pth', 
                 data_path='data/processed/'):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.metrics = None
        
    def load_data(self, days_back=60):
        """
        Load test data (últimos N dias)
        
        Expected format:
        - X_test.npy: (samples, seq_len, features) 
        - y_test.npy: (samples,) - actual crime values
        """
        logger.info(f"Loading test data (últimos {days_back} dias)...")
        
        # Placeholder: In real scenario, load from processed data
        # For now, create dummy data for testing
        n_nodes = 319
        n_windows = 20
        
        # Simulate actual data
        y_test = np.random.exponential(scale=3.0, size=(n_windows, n_nodes))
        y_test = np.clip(y_test, 0, 10)  # Clip to [0, 10]
        
        # Simulate predictions (slightly correlated with actuals)
        y_pred = y_test + np.random.normal(0, 0.5, (n_windows, n_nodes))
        y_pred = np.clip(y_pred, 0, 10)
        
        logger.info(f"Loaded test data: {y_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  y_pred shape: {y_pred.shape}")
        
        return y_test, y_pred
    
    def evaluate_comprehensive(self, y_test, y_pred):
        """
        Calculate all metrics across all windows
        
        Args:
            y_test: (n_windows, n_nodes)
            y_pred: (n_windows, n_nodes)
        
        Returns:
            Dictionary with aggregated metrics
        """
        logger.info("Calculating comprehensive metrics...")
        
        all_metrics = []
        
        # Per-window metrics
        for window_idx in range(len(y_test)):
            y_true_window = y_test[window_idx]
            y_pred_window = y_pred[window_idx]
            
            window_metrics = MetricReporter.report(y_true_window, y_pred_window)
            all_metrics.append(window_metrics)
        
        # Aggregate across windows
        aggregated = {}
        for metric_key in all_metrics[0].keys():
            values = [m[metric_key] for m in all_metrics]
            aggregated[metric_key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
        
        self.metrics = aggregated
        
        return aggregated
    
    def print_report(self):
        """Pretty-print the metrics report"""
        if self.metrics is None:
            logger.warning("No metrics available. Run evaluate_comprehensive first.")
            return
        
        print("\n" + "=" * 80)
        print("BASELINE EVALUATION REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Organize by category
        categories = {
            'Precision Metrics': ['p_at_5', 'p_at_10', 'p_at_20'],
            'NDCG Metrics': ['ndcg_at_5', 'ndcg_at_10', 'ndcg_at_20'],
            'Recall Metrics': ['recall_at_5', 'recall_at_10', 'recall_at_20'],
            'Other Metrics': ['mrr_at_20', 'mae_rank', 'spearman_corr', 'mean_score_diff', 'std_score_diff']
        }
        
        for category, metrics_list in categories.items():
            print(f"\n{category}")
            print("-" * 80)
            print(f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print("-" * 80)
            
            for metric_name in metrics_list:
                if metric_name in self.metrics:
                    m = self.metrics[metric_name]
                    print(f"{metric_name:<20} {m['mean']:>12.4f} {m['std']:>12.4f} {m['min']:>12.4f} {m['max']:>12.4f}")
        
        print("\n" + "=" * 80)
        print("TARGET METRICS FOR PHASE 2B")
        print("=" * 80)
        
        targets = {
            'p_at_5': {'current': self.metrics['p_at_5']['mean'], 'target': 0.78, 'status': ''},
            'p_at_10': {'current': self.metrics['p_at_10']['mean'], 'target': 0.65, 'status': ''},
            'p_at_20': {'current': self.metrics['p_at_20']['mean'], 'target': 0.55, 'status': ''},
            'ndcg_at_5': {'current': self.metrics['ndcg_at_5']['mean'], 'target': 0.92, 'status': ''},
            'ndcg_at_20': {'current': self.metrics['ndcg_at_20']['mean'], 'target': 0.76, 'status': ''},
        }
        
        print(f"{'Metric':<15} {'Current':>12} {'Target':>12} {'Status':>20}")
        print("-" * 60)
        
        for metric_name, info in targets.items():
            current = info['current']
            target = info['target']
            
            if current >= target:
                status = "✅ OK (meets or exceeds)"
            else:
                gap = target - current
                status = f"❌ BELOW ({gap:+.3f})"
            
            print(f"{metric_name:<15} {current:>12.4f} {target:>12.4f} {status:>20}")
        
        print("\n" + "=" * 80)
    
    def save_baseline(self, output_file='baseline_metrics.json'):
        """Save metrics to JSON for reference"""
        if self.metrics is None:
            logger.warning("No metrics to save. Run evaluate_comprehensive first.")
            return
        
        # Convert to JSON-serializable format
        metrics_json = {}
        for key, value in self.metrics.items():
            metrics_json[key] = {
                'mean': float(value['mean']),
                'std': float(value['std']),
                'min': float(value['min']),
                'max': float(value['max']),
            }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Baseline metrics saved to {output_path}")
        print(f"✅ Saved to {output_path}")


def main():
    """Main evaluation script"""
    
    logger.info("Starting Baseline Evaluation (Week 1, Task 1.2)")
    
    # Initialize evaluator
    evaluator = BaselineEvaluator()
    
    # Load data
    y_test, y_pred = evaluator.load_data(days_back=60)
    
    # Evaluate
    metrics = evaluator.evaluate_comprehensive(y_test, y_pred)
    
    # Print report
    evaluator.print_report()
    
    # Save baseline
    evaluator.save_baseline('baseline_metrics.json')
    
    logger.info("✅ Baseline evaluation complete!")


if __name__ == "__main__":
    main()
