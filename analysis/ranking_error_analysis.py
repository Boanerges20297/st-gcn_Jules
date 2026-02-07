"""
Ranking Error Analysis
Analyzes where and why the model makes mistakes in ranking

Author: ST-GCN Enhanced System
Date: Feb 2026

Analysis:
- Identify undershooting vs overshooting errors
- Characterize consistently missed nodes
- Find patterns in errors
- Suggest improvements
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metrics import MetricReporter


class RankingErrorAnalyzer:
    """
    Comprehensive ranking error analysis
    
    Identifies:
    - Individual node ranking errors
    - Error patterns across windows
    - Systematic biases
    - Nodes that are consistently missed
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.all_errors = []
        self.node_error_history = {}
    
    def analyze_single_window(self, y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             window_idx: int = 0) -> Dict:
        """
        Analyze ranking errors for a single window/sample
        
        Args:
            y_true: Actual scores (n_nodes,)
            y_pred: Predicted scores (n_nodes,)
            window_idx: Window identifier
        
        Returns:
            Dictionary with detailed error analysis
        """
        n_nodes = len(y_true)
        
        # Get rankings
        true_ranks = np.argsort(-y_true)
        pred_ranks = np.argsort(-y_pred)
        
        # Create rank position lookup
        rank_positions_true = np.empty(n_nodes, dtype=int)
        rank_positions_pred = np.empty(n_nodes, dtype=int)
        
        for pos, node_id in enumerate(true_ranks):
            rank_positions_true[node_id] = pos
        
        for pos, node_id in enumerate(pred_ranks):
            rank_positions_pred[node_id] = pos
        
        # Calculate errors per node
        node_errors = []
        
        for node_id in range(n_nodes):
            true_rank = rank_positions_true[node_id]
            pred_rank = rank_positions_pred[node_id]
            true_score = y_true[node_id]
            pred_score = y_pred[node_id]
            
            rank_error = pred_rank - true_rank  # Positive = overshooting, Negative = undershooting
            score_error = pred_score - true_score
            
            # Determine error type
            if rank_error > 0:
                error_type = 'overshooting'  # Predicted rank too low (bad)
            elif rank_error < 0:
                error_type = 'undershooting'  # Predicted rank too high (over-optimistic)
            else:
                error_type = 'correct'
            
            # Determine importance tier
            if true_rank < 5:
                tier = 'top_5'
            elif true_rank < 20:
                tier = 'long_tail_20'
            elif true_rank < 50:
                tier = 'long_tail_50'
            else:
                tier = 'tail'
            
            node_errors.append({
                'node_id': int(node_id),
                'true_rank': int(true_rank),
                'pred_rank': int(pred_rank),
                'rank_error': int(rank_error),
                'true_score': float(true_score),
                'pred_score': float(pred_score),
                'score_error': float(score_error),
                'error_type': error_type,
                'tier': tier,
                'abs_rank_error': abs(rank_error)
            })
        
        # Aggregate statistics
        error_counts = {}
        for error in node_errors:
            et = error['error_type']
            error_counts[et] = error_counts.get(et, 0) + 1
        
        tier_errors = {}
        for error in node_errors:
            tier = error['tier']
            if tier not in tier_errors:
                tier_errors[tier] = {'undershooting': 0, 'overshooting': 0, 'correct': 0}
            tier_errors[tier][error['error_type']] += 1
        
        # Find worst errors
        worst_errors = sorted(node_errors, key=lambda x: x['abs_rank_error'], reverse=True)[:10]
        
        # Metrics
        p5 = MetricReporter.precision_at_k(y_true, y_pred, 5)
        p20 = MetricReporter.precision_at_k(y_true, y_pred, 20)
        ndcg20 = MetricReporter.ndcg_at_k(y_true, y_pred, 20)
        
        analysis = {
            'window_idx': window_idx,
            'total_nodes': n_nodes,
            'error_counts': error_counts,
            'tier_errors': tier_errors,
            'node_errors': node_errors,
            'worst_errors': worst_errors,
            'metrics': {
                'p_at_5': float(p5),
                'p_at_20': float(p20),
                'ndcg_at_20': float(ndcg20),
                'mean_abs_rank_error': float(np.mean([e['abs_rank_error'] for e in node_errors])),
                'median_rank_error': float(np.median([e['rank_error'] for e in node_errors])),
                'std_rank_error': float(np.std([e['rank_error'] for e in node_errors]))
            }
        }
        
        return analysis
    
    def analyze_multi_window(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze errors across multiple windows
        
        Args:
            y_test: Test data (n_windows, n_nodes)
            y_pred: Predictions (n_windows, n_nodes)
        
        Returns:
            Aggregated analysis
        """
        logger.info(f"Analyzing {len(y_test)} windows...")
        
        all_analyses = []
        node_error_frequency = {}  # node_id -> count of times in error
        node_error_magnitude = {}  # node_id -> sum of error magnitudes
        
        for window_idx in range(len(y_test)):
            analysis = self.analyze_single_window(
                y_test[window_idx], 
                y_pred[window_idx], 
                window_idx
            )
            all_analyses.append(analysis)
            
            # Accumulate node-level stats
            for error in analysis['node_errors']:
                node_id = error['node_id']
                if node_id not in node_error_frequency:
                    node_error_frequency[node_id] = 0
                    node_error_magnitude[node_id] = 0.0
                
                if error['error_type'] != 'correct':
                    node_error_frequency[node_id] += 1
                    node_error_magnitude[node_id] += error['abs_rank_error']
        
        # Calculate node-level statistics
        problem_nodes = []
        for node_id in sorted(node_error_frequency.keys()):
            freq = node_error_frequency[node_id]
            if freq > 0:
                problem_nodes.append({
                    'node_id': int(node_id),
                    'error_frequency': int(freq),
                    'error_rate': freq / len(all_analyses),
                    'avg_error_magnitude': node_error_magnitude[node_id] / freq
                })
        
        # Sort by frequency
        problem_nodes.sort(key=lambda x: x['error_rate'], reverse=True)
        
        # Aggregate metrics
        aggregated = {
            'total_windows': len(all_analyses),
            'window_analyses': all_analyses,
            'most_problematic_nodes': problem_nodes[:20],
            'aggregated_metrics': {
                'avg_p_at_5': float(np.mean([a['metrics']['p_at_5'] for a in all_analyses])),
                'avg_p_at_20': float(np.mean([a['metrics']['p_at_20'] for a in all_analyses])),
                'avg_ndcg_at_20': float(np.mean([a['metrics']['ndcg_at_20'] for a in all_analyses])),
                'mean_abs_rank_error': float(np.mean([a['metrics']['mean_abs_rank_error'] for a in all_analyses])),
                'undershooting_total': sum(
                    a['error_counts'].get('undershooting', 0) for a in all_analyses
                ),
                'overshooting_total': sum(
                    a['error_counts'].get('overshooting', 0) for a in all_analyses
                ),
            }
        }
        
        return aggregated
    
    def identify_patterns(self, analysis: Dict) -> Dict:
        """
        Identify patterns in errors
        
        Args:
            analysis: Output from analyze_multi_window
        
        Returns:
            Dictionary with pattern insights
        """
        logger.info("Identifying error patterns...")
        
        patterns = {
            'error_types': {},
            'tier_analysis': {},
            'recommendations': []
        }
        
        # Error type analysis
        total_undershot = analysis['aggregated_metrics']['undershooting_total']
        total_overshot = analysis['aggregated_metrics']['overshooting_total']
        total_errors = total_undershot + total_overshot
        
        patterns['error_types'] = {
            'undershooting_rate': total_undershot / (total_errors + 1),
            'overshooting_rate': total_overshot / (total_errors + 1),
            'primary_error_type': 'undershooting' if total_undershot > total_overshot else 'overshooting'
        }
        
        # Tier analysis  
        tier_names = ['top_5', 'long_tail_20', 'long_tail_50', 'tail']
        tier_error_rates = {}
        
        for window_analysis in analysis['window_analyses'][:1]:  # Use first window as representative
            for tier, counts in window_analysis['tier_errors'].items():
                total_tier = sum(counts.values())
                error_rate = (counts['undershooting'] + counts['overshooting']) / (total_tier + 1)
                tier_error_rates[tier] = error_rate
        
        patterns['tier_analysis'] = tier_error_rates
        
        # Generate recommendations
        if patterns['error_types']['primary_error_type'] == 'undershooting':
            patterns['recommendations'].append(
                "Model tends to predict ranks too high (UNDERSHOOTING). "
                "Targets are overconfident. Consider reducing learning rate or using regularization."
            )
        else:
            patterns['recommendations'].append(
                "Model tends to predict ranks too low (OVERSHOOTING). "
                "Targets lack confidence. Consider using different loss weight or feature engineering."
            )
        
        if analysis['aggregated_metrics']['avg_p_at_20'] < 0.55:
            patterns['recommendations'].append(
                "P@20 is below target (0.55). Use TopKLoss or CombinedLoss to emphasize long-tail."
            )
        
        if len(analysis['most_problematic_nodes']) > 0:
            top_problem = analysis['most_problematic_nodes'][0]
            patterns['recommendations'].append(
                f"Node {top_problem['node_id']} is consistently problematic (error rate={top_problem['error_rate']:.1%}). "
                f"May need feature engineering for this specific node."
            )
        
        return patterns
    
    def print_report(self, analysis: Dict, patterns: Dict):
        """Pretty-print analysis report"""
        
        print("\n" + "=" * 80)
        print("RANKING ERROR ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\nAggregated Metrics Across {analysis['total_windows']} Windows:")
        print("-" * 80)
        agg = analysis['aggregated_metrics']
        print(f"  P@5:  {agg['avg_p_at_5']:.4f}")
        print(f"  P@20: {agg['avg_p_at_20']:.4f}")
        print(f"  NDCG@20: {agg['avg_ndcg_at_20']:.4f}")
        print(f"  Mean absolute rank error: {agg['mean_abs_rank_error']:.2f}")
        print(f"  Undershooting errors: {agg['undershooting_total']}")
        print(f"  Overshooting errors: {agg['overshooting_total']}")
        
        print(f"\nError Type Analysis:")
        print("-" * 80)
        err = patterns['error_types']
        print(f"  Undershooting rate: {err['undershooting_rate']:.1%}")
        print(f"  Overshooting rate: {err['overshooting_rate']:.1%}")
        print(f"  Primary error type: {err['primary_error_type'].upper()}")
        
        print(f"\nMost Problematic Nodes:")
        print("-" * 80)
        for i, node in enumerate(analysis['most_problematic_nodes'][:5], 1):
            print(f"  {i}. Node {node['node_id']:3d} - "
                  f"Error rate: {node['error_rate']:.1%}, "
                  f"Avg magnitude: {node['avg_error_magnitude']:.1f}")
        
        print(f"\nRecommendations:")
        print("-" * 80)
        for i, rec in enumerate(patterns['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
    
    def save_analysis(self, analysis: Dict, output_file: str = 'ranking_error_analysis.json'):
        """Save analysis to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {output_path}")


def main():
    """Main analysis script"""
    
    logger.info("Starting Ranking Error Analysis (Week 3, Task 3.1)")
    
    # Initialize analyzer
    analyzer = RankingErrorAnalyzer()
    
    # Generate dummy test data
    logger.info("Generating test data...")
    n_windows = 20
    n_nodes = 319
    
    y_test = np.random.exponential(scale=3.0, size=(n_windows, n_nodes))
    y_test = np.clip(y_test, 0, 10)
    
    y_pred = y_test + np.random.normal(0, 0.8, (n_windows, n_nodes))
    y_pred = np.clip(y_pred, 0, 10)
    
    # Run analysis
    analysis = analyzer.analyze_multi_window(y_test, y_pred)
    patterns = analyzer.identify_patterns(analysis)
    
    # Print report
    analyzer.print_report(analysis, patterns)
    
    # Save analysis
    analyzer.save_analysis(analysis, 'ranking_error_analysis.json')
    
    logger.info("✅ Error analysis complete!")


if __name__ == "__main__":
    main()
