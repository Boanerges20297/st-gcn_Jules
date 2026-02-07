"""
Long-Tail Analysis Script
Identifies which nodes should be in top-20 but are being missed by the model

Author: ST-GCN Enhanced System
Date: Feb 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LongTailAnalyzer:
    """
    Analyzes ranking errors for nodes outside top-5
    
    Questions answered:
    - Which nodes should be in top-20 but aren't?
    - Why are they being missed?
    - Temporal? Spatial? Scale issues?
    """
    
    def __init__(self, data_path='data/processed/'):
        self.data_path = Path(data_path)
        
    def analyze_ranking_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               k: int = 20) -> Dict:
        """
        Identify nodes in top-K real but not in top-K predicted
        
        Args:
            y_true: Actual scores (shape: n_nodes,)
            y_pred: Predicted scores (shape: n_nodes,)
            k: K for top-K
        
        Returns:
            Analysis dictionary
        """
        logger.info(f"Analyzing ranking errors for top-{k}...")
        
        # Get indices
        true_top_k_indices = np.argsort(-y_true)[:k]
        pred_top_k_indices = np.argsort(-y_pred)[:k]
        
        true_top_k_set = set(true_top_k_indices)
        pred_top_k_set = set(pred_top_k_indices)
        
        # Classification
        correct = true_top_k_set & pred_top_k_set
        missed = true_top_k_set - pred_top_k_set      # Should be in top-K but not
        false_positives = pred_top_k_set - true_top_k_set  # Not in top-K but predicted
        
        logger.info(f"Correct: {len(correct)}/{k}")
        logger.info(f"Missed:  {len(missed)}/{k}")
        logger.info(f"False positives: {len(false_positives)}/{k}")
        
        # Detailed analysis of missed nodes
        missed_details = []
        for node_idx in sorted(missed):
            true_rank = np.where(np.argsort(-y_true) == node_idx)[0][0]
            pred_rank = np.where(np.argsort(-y_pred) == node_idx)[0][0]
            true_score = y_true[node_idx]
            pred_score = y_pred[node_idx]
            
            missed_details.append({
                'node_id': int(node_idx),
                'true_rank': int(true_rank) + 1,
                'pred_rank': int(pred_rank) + 1,
                'true_score': float(true_score),
                'pred_score': float(pred_score),
                'score_error': float(true_score - pred_score),
                'rank_error': int(pred_rank - true_rank),
                'error_type': 'undershooting' if pred_rank > true_rank else 'overshooting'
            })
        
        # Analysis summary
        analysis = {
            'window_summary': {
                'total_nodes': len(y_true),
                'top_k': k,
                'correct_in_top_k': len(correct),
                'missed_from_top_k': len(missed),
                'false_positives': len(false_positives),
                'p_at_k': len(correct) / k,
                'recall_at_k': len(correct) / k,
            },
            'correct_nodes': sorted(list(correct)),
            'missed_nodes_detailed': missed_details,
            'false_positive_detailed': [
                {
                    'node_id': int(node_idx),
                    'true_rank': int(np.where(np.argsort(-y_true) == node_idx)[0][0]) + 1,
                    'pred_rank': int(np.where(np.argsort(-y_pred) == node_idx)[0][0]) + 1,
                    'true_score': float(y_true[node_idx]),
                    'pred_score': float(y_pred[node_idx]),
                }
                for node_idx in sorted(false_positives)
            ]
        }
        
        return analysis
    
    def analyze_error_patterns(self, errors_list: List[Dict]) -> Dict:
        """
        Find patterns in errors across multiple windows
        
        Args:
            errors_list: List of error analyses from multiple windows
        
        Returns:
            Pattern summary
        """
        logger.info("Analyzing error patterns across windows...")
        
        # Aggregate by node
        node_errors = {}
        for window_result in errors_list:
            for missed_node in window_result['missed_nodes_detailed']:
                node_id = missed_node['node_id']
                if node_id not in node_errors:
                    node_errors[node_id] = []
                node_errors[node_id].append(missed_node)
        
        # Find consistently missed nodes
        consistently_missed = []
        for node_id, errors in node_errors.items():
            miss_rate = len(errors) / len(errors_list)
            avg_rank_error = np.mean([e['rank_error'] for e in errors])
            error_type_counts = {}
            for e in errors:
                et = e['error_type']
                error_type_counts[et] = error_type_counts.get(et, 0) + 1
            
            consistently_missed.append({
                'node_id': node_id,
                'miss_rate': miss_rate,
                'times_missed': len(errors),
                'avg_rank_error': float(avg_rank_error),
                'primary_error_type': max(error_type_counts, key=error_type_counts.get),
                'error_distribution': error_type_counts
            })
        
        # Sort by miss rate
        consistently_missed.sort(key=lambda x: x['miss_rate'], reverse=True)
        
        return {
            'total_nodes': len(node_errors),
            'consistently_missed_top_10': consistently_missed[:10],
            'all_consistently_missed': consistently_missed
        }
    
    def generate_report(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Generate complete long-tail analysis report
        
        Args:
            y_test: (n_windows, n_nodes)
            y_pred: (n_windows, n_nodes)
        
        Returns:
            Complete analysis report
        """
        logger.info("Generating long-tail analysis report...")
        
        all_analyses = []
        
        # Analyze each window
        for window_idx in range(len(y_test)):
            window_analysis = self.analyze_ranking_errors(
                y_test[window_idx], 
                y_pred[window_idx], 
                k=20
            )
            all_analyses.append(window_analysis)
        
        # Aggregate patterns
        aggregate_summary = {
            'total_windows': len(all_analyses),
            'avg_p_at_20': np.mean([a['window_summary']['p_at_k'] for a in all_analyses]),
            'avg_recall_at_20': np.mean([a['window_summary']['recall_at_k'] for a in all_analyses]),
            'window_results': all_analyses
        }
        
        # Pattern analysis
        pattern_analysis = self.analyze_error_patterns(all_analyses)
        
        report = {
            'timestamp': str(np.datetime64('today')),
            'summary': aggregate_summary,
            'patterns': pattern_analysis,
            'recommendations': self._generate_recommendations(aggregate_summary, pattern_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict, patterns: Dict) -> List[str]:
        """
        Generate recommendations based on analysis
        
        Args:
            summary: Aggregated summary
            patterns: Error patterns
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        avg_p20 = summary['avg_p_at_20']
        
        if avg_p20 < 0.50:
            recommendations.append(
                "P@20 is critically low (<50%). Consider reviewing model architecture or features."
            )
        elif avg_p20 < 0.55:
            recommendations.append(
                "P@20 is below target (0.55). Focus on long-tail optimization with weighted loss."
            )
        else:
            recommendations.append(
                "P@20 is acceptable. Continue with current approach."
            )
        
        # Check for systematic patterns
        primarily_undershot = sum(
            1 for node in patterns['all_consistently_missed'] 
            if node['primary_error_type'] == 'undershooting'
        )
        
        if primarily_undershot > len(patterns['all_consistently_missed']) / 2:
            recommendations.append(
                "Most errors are undershooting (predicting rank too low). "
                "Model may have low recall on important nodes."
            )
        
        top_missed = patterns['consistently_missed_top_10']
        if top_missed:
            recommendations.append(
                f"Top 3 consistently missed nodes: {', '.join([str(n['node_id']) for n in top_missed[:3]])}. "
                f"Investigate why model struggles with these nodes."
            )
        
        return recommendations
    
    def save_report(self, report: Dict, output_file='long_tail_analysis.json'):
        """Save analysis to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        report_serializable = json.loads(json.dumps(report, default=str))
        
        with open(output_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        print(f"✅ Report saved to {output_path}")
    
    def print_report(self, report: Dict):
        """Pretty-print the analysis report"""
        print("\n" + "=" * 80)
        print("LONG-TAIL ANALYSIS REPORT")
        print("=" * 80)
        
        summary = report['summary']
        print(f"\nWindow Summary:")
        print(f"  Total windows analyzed: {summary['total_windows']}")
        print(f"  Average P@20: {summary['avg_p_at_20']:.4f}")
        print(f"  Average Recall@20: {summary['avg_recall_at_20']:.4f}")
        
        patterns = report['patterns']
        print(f"\nError Patterns:")
        print(f"  Total nodes with errors: {patterns['total_nodes']}")
        print(f"  Top 10 consistently missed nodes:")
        
        for i, node in enumerate(patterns['consistently_missed_top_10'], 1):
            print(f"\n    {i}. Node {node['node_id']}")
            print(f"       Miss rate: {node['miss_rate']:.1%} ({node['times_missed']} times)")
            print(f"       Avg rank error: {node['avg_rank_error']:.1f}")
            print(f"       Primary error: {node['primary_error_type']}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)


def main():
    """Main analysis script"""
    
    logger.info("Starting Long-Tail Analysis (Week 1, Task 1.3)")
    
    # Initialize analyzer
    analyzer = LongTailAnalyzer()
    
    # Generate dummy test data
    logger.info("Generating test data...")
    n_windows = 20
    n_nodes = 319
    
    y_test = np.random.exponential(scale=3.0, size=(n_windows, n_nodes))
    y_test = np.clip(y_test, 0, 10)
    
    y_pred = y_test + np.random.normal(0, 0.7, (n_windows, n_nodes))
    y_pred = np.clip(y_pred, 0, 10)
    
    # Run analysis
    report = analyzer.generate_report(y_test, y_pred)
    
    # Print report
    analyzer.print_report(report)
    
    # Save report
    analyzer.save_report(report, 'long_tail_analysis.json')
    
    logger.info("✅ Long-tail analysis complete!")


if __name__ == "__main__":
    main()
