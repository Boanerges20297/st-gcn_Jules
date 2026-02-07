"""
Model Comparison and Selection
Compares different trained models and selects the best one

Author: ST-GCN Enhanced System
Date: Feb 2026

Comparison criteria:
- P@5, P@10, P@20, NDCG@K
- Generalization (train vs val gap)
- Computational efficiency
- Overall ranking quality

Selection criteria:
- P@5 ≥ 0.78 (don't hurt existing performance)
- P@20 ≥ 0.55 (improve long-tail)
- Generalization gap < 10%
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics import MetricReporter


class ModelComparator:
    """
    Compares multiple ranking models
    
    Evaluates on multiple metrics and provides selection guidance
    """
    
    def __init__(self):
        """Initialize comparator"""
        self.models = {}  # name -> metrics dict
        self.selection_criteria = {
            'p_at_5_min': 0.78,
            'p_at_20_min': 0.55,
            'ndcg_at_5_min': 0.92,
            'generalization_gap_max': 0.10
        }
    
    def register_model(self, model_name: str, 
                      train_metrics: Dict, 
                      val_metrics: Dict,
                      model_path: str = None,
                      notes: str = ""):
        """
        Register a model for comparison
        
        Args:
            model_name: Human-readable name
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            model_path: Path to model file
            notes: Additional notes
        """
        self.models[model_name] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_path': model_path,
            'notes': notes,
            'scores': {}  # Will be populated during evaluation
        }
        
        logger.info(f"Registered model: {model_name}")
    
    def evaluate_model(self, model_name: str):
        """
        Evaluate a single model against criteria
        
        Args:
            model_name: Model to evaluate
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not registered")
            return
        
        model_data = self.models[model_name]
        val_metrics = model_data['val_metrics']
        
        scores = {}
        
        # P@K scores
        scores['p_at_5'] = val_metrics.get('p_at_5', 0)
        scores['p_at_20'] = val_metrics.get('p_at_20', 0)
        scores['ndcg_at_5'] = val_metrics.get('ndcg_at_5', 0)
        scores['ndcg_at_20'] = val_metrics.get('ndcg_at_20', 0)
        
        # Generalization (train vs val gap)
        p5_gap = abs(
            model_data['train_metrics'].get('p_at_5', 0) - 
            val_metrics.get('p_at_5', 0)
        )
        p20_gap = abs(
            model_data['train_metrics'].get('p_at_20', 0) - 
            val_metrics.get('p_at_20', 0)
        )
        scores['generalization_gap'] = max(p5_gap, p20_gap)
        
        # Overall quality score (weighted)
        scores['overall_score'] = (
            0.25 * scores['p_at_5'] +
            0.35 * scores['p_at_20'] +
            0.25 * scores['ndcg_at_20'] +
            0.15 * (1 - scores['generalization_gap'])  # Lower gap is better
        )
        
        model_data['scores'] = scores
    
    def evaluate_all(self):
        """Evaluate all registered models"""
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
    
    def check_criteria(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Check if model meets selection criteria
        
        Args:
            model_name: Model to check
        
        Returns:
            Tuple of (passes_all_criteria, list_of_failures)
        """
        if model_name not in self.models:
            return False, ["Model not registered"]
        
        model_data = self.models[model_name]
        val_metrics = model_data['val_metrics']
        scores = model_data['scores']
        
        failures = []
        
        # Check each criterion
        if val_metrics.get('p_at_5', 0) < self.selection_criteria['p_at_5_min']:
            gap = self.selection_criteria['p_at_5_min'] - val_metrics['p_at_5']
            failures.append(f"P@5 too low ({val_metrics['p_at_5']:.4f} < {self.selection_criteria['p_at_5_min']:.4f})")
        
        if val_metrics.get('p_at_20', 0) < self.selection_criteria['p_at_20_min']:
            gap = self.selection_criteria['p_at_20_min'] - val_metrics['p_at_20']
            failures.append(f"P@20 too low ({val_metrics['p_at_20']:.4f} < {self.selection_criteria['p_at_20_min']:.4f})")
        
        if val_metrics.get('ndcg_at_5', 0) < self.selection_criteria['ndcg_at_5_min']:
            failures.append(f"NDCG@5 too low ({val_metrics['ndcg_at_5']:.4f} < {self.selection_criteria['ndcg_at_5_min']:.4f})")
        
        if scores.get('generalization_gap', 1.0) > self.selection_criteria['generalization_gap_max']:
            failures.append(f"Generalization gap too high ({scores['generalization_gap']:.1%})")
        
        passes = len(failures) == 0
        return passes, failures
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """
        Get best model according to overall score
        
        Returns:
            Tuple of (model_name, model_data)
        """
        best_name = None
        best_score = -1
        
        for model_name, model_data in self.models.items():
            score = model_data['scores'].get('overall_score', 0)
            if score > best_score:
                best_score = score
                best_name = model_name
        
        if best_name:
            return best_name, self.models[best_name]
        else:
            return None, None
    
    def print_comparison_table(self):
        """Print comparison table"""
        if not self.models:
            print("No models registered for comparison")
            return
        
        print("\n" + "=" * 120)
        print("MODEL COMPARISON TABLE")
        print("=" * 120)
        
        # Header
        print(f"{'Model':<20} {'P@5':>8} {'P@20':>8} {'NDCG@5':>8} {'NDCG@20':>8} "
              f"{'GenGap':>8} {'Score':>8} {'Status':>10}")
        print("-" * 120)
        
        # Models
        for model_name, model_data in self.models.items():
            val_metrics = model_data['val_metrics']
            scores = model_data['scores']
            
            passes, _ = self.check_criteria(model_name)
            status = "[PASS]" if passes else "[FAIL]"
            
            print(f"{model_name:<20} "
                  f"{val_metrics.get('p_at_5', 0):>8.4f} "
                  f"{val_metrics.get('p_at_20', 0):>8.4f} "
                  f"{val_metrics.get('ndcg_at_5', 0):>8.4f} "
                  f"{val_metrics.get('ndcg_at_20', 0):>8.4f} "
                  f"{scores.get('generalization_gap', 0):>8.1%} "
                  f"{scores.get('overall_score', 0):>8.4f} "
                  f"{status:>10}")
        
        print("=" * 120)
    
    def print_detailed_report(self):
        """Print detailed report for each model"""
        
        print("\n" + "=" * 80)
        print("DETAILED MODEL EVALUATION")
        print("=" * 80)
        
        for model_name, model_data in self.models.items():
            val_metrics = model_data['val_metrics']
            scores = model_data['scores']
            passes, failures = self.check_criteria(model_name)
            
            print(f"\n{model_name}")
            print("-" * 80)
            
            if model_data['notes']:
                print(f"Notes: {model_data['notes']}")
            
            print(f"Validation Metrics:")
            print(f"  P@5:  {val_metrics.get('p_at_5', 0):.4f}")
            print(f"  P@10: {val_metrics.get('p_at_10', 0):.4f}")
            print(f"  P@20: {val_metrics.get('p_at_20', 0):.4f}")
            print(f"  NDCG@5:  {val_metrics.get('ndcg_at_5', 0):.4f}")
            print(f"  NDCG@20: {val_metrics.get('ndcg_at_20', 0):.4f}")
            
            print(f"Quality Scores:")
            print(f"  Generalization gap: {scores.get('generalization_gap', 0):.1%}")
            print(f"  Overall score: {scores.get('overall_score', 0):.4f}")
            
            if passes:
                print(f"Status: [PASS] PASSES all criteria")
            else:
                print(f"Status: [FAIL]")
                for failure in failures:
                    print(f"  - {failure}")
    
    def print_recommendation(self):
        """Print recommendation"""
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        
        best_name, best_data = self.get_best_model()
        
        if best_name:
            passes, failures = self.check_criteria(best_name)
            
            print(f"\nBest model: {best_name}")
            print(f"Overall score: {best_data['scores'].get('overall_score', 0):.4f}")
            
            if passes:
                print(f"\n✅ RECOMMENDATION: Deploy {best_name}")
                print(f"\nRationale:")
                print(f"  - Meets all selection criteria")
                print(f"  - Highest overall score")
                print(f"  - Ready for production")
                
                if best_data['model_path']:
                    print(f"\nModel path: {best_data['model_path']}")
            else:
                print(f"\n[WARNING] Best model fails some criteria")
                print(f"Failures:")
                for failure in failures:
                    print(f"  - {failure}")
                
                print(f"\nOptions:")
                print(f"  1. Continue training with different loss")
                print(f"  2. Adjust selection criteria")
                print(f"  3. Collect more data")
        else:
            print("No models to recommend")
    
    def save_comparison(self, output_file: str = 'model_comparison.json'):
        """Save comparison results"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare JSON-serializable data
        comparison_data = {}
        for model_name, model_data in self.models.items():
            comparison_data[model_name] = {
                'val_metrics': model_data['val_metrics'],
                'scores': model_data['scores'],
                'notes': model_data['notes'],
                'passes_criteria': self.check_criteria(model_name)[0]
            }
        
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Comparison saved to {output_path}")


def main():
    """Main comparison script"""
    
    logger.info("Starting Model Comparison (Week 3, Task 3.4)")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Register models (simulated results)
    logger.info("Registering models for comparison...")
    
    # Model 1: Baseline (Week 2)
    comparator.register_model(
        "baseline_with_anomaly",
        train_metrics={
            'p_at_5': 0.82, 'p_at_10': 0.70, 'p_at_20': 0.50,
            'ndcg_at_5': 0.93, 'ndcg_at_20': 0.76
        },
        val_metrics={
            'p_at_5': 0.80, 'p_at_10': 0.68, 'p_at_20': 0.48,
            'ndcg_at_5': 0.92, 'ndcg_at_20': 0.74
        },
        model_path='models/ranking_model_with_anomaly.pth',
        notes='Week 2 baseline with event integration'
    )
    
    # Model 2: P@5 focused (with margin)
    comparator.register_model(
        "model_p5_focused",
        train_metrics={
            'p_at_5': 0.85, 'p_at_10': 0.68, 'p_at_20': 0.45,
            'ndcg_at_5': 0.94, 'ndcg_at_20': 0.73
        },
        val_metrics={
            'p_at_5': 0.83, 'p_at_10': 0.67, 'p_at_20': 0.44,
            'ndcg_at_5': 0.93, 'ndcg_at_20': 0.72
        },
        model_path='models/model_p5_focused.pth',
        notes='Optimized for P@5, may hurt long-tail'
    )
    
    # Model 3: P@20 optimized (combined loss)
    comparator.register_model(
        "model_with_p20_optimization",
        train_metrics={
            'p_at_5': 0.80, 'p_at_10': 0.72, 'p_at_20': 0.57,
            'ndcg_at_5': 0.91, 'ndcg_at_20': 0.78
        },
        val_metrics={
            'p_at_5': 0.78, 'p_at_10': 0.70, 'p_at_20': 0.55,
            'ndcg_at_5': 0.90, 'ndcg_at_20': 0.77
        },
        model_path='models/model_with_p20_optimization.pth',
        notes='Combined loss (0.5 P@5 + 0.5 P@20), balanced approach'
    )
    
    # Model 4: Ensemble (not standard)
    comparator.register_model(
        "model_balanced_loss",
        train_metrics={
            'p_at_5': 0.81, 'p_at_10': 0.71, 'p_at_20': 0.56,
            'ndcg_at_5': 0.92, 'ndcg_at_20': 0.77
        },
        val_metrics={
            'p_at_5': 0.79, 'p_at_10': 0.69, 'p_at_20': 0.54,
            'ndcg_at_5': 0.91, 'ndcg_at_20': 0.76
        },
        model_path='models/model_balanced_loss.pth',
        notes='Alpha=0.6, slightly more weight on P@5'
    )
    
    # Evaluate all models
    logger.info("Evaluating models...")
    comparator.evaluate_all()
    
    # Print reports
    comparator.print_comparison_table()
    comparator.print_detailed_report()
    comparator.print_recommendation()
    
    # Save comparison
    comparator.save_comparison('model_comparison.json')
    
    logger.info("✅ Comparison complete!")


if __name__ == "__main__":
    main()
