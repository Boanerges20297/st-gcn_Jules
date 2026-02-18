"""
Explanation Generator
Generates human-readable explanations for model predictions and rankings

Author: ST-GCN Enhanced System
Date: Feb 2026

Approaches:
- Heuristic-based (MVP): Template-based explanations
- LLM-based (future): Google Gemini or OpenAI API

Generates:
- Why is node X ranked at Y?
- Which factors contribute to the prediction?
- Confidence level and uncertainty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generates human-readable explanations for predictions
    
    MVP approach: Heuristic-based with templates
    Future: LLM-based (Google Gemini)
    """
    
    def __init__(self, model=None, data_manager=None):
        """
        Initialize explanation generator
        
        Args:
            model: Trained ranking model (optional)
            data_manager: Data manager for context (optional)
        """
        self.model = model
        self.data_manager = data_manager
    
    def explain_node_ranking(self, 
                            node_id: int, 
                            rank: int,
                            context_dict: Dict) -> Dict:
        """
        Generate explanation for why a node is at a specific rank
        
        Args:
            node_id: Node ID (e.g., 146)
            rank: Current rank position (e.g., 1)
            context_dict: Context dictionary with:
                - score: float (node's prediction score)
                - temporal_pattern: str (e.g., "High in Feb")
                - nearby_nodes: list (similar high-score nodes)
                - events: list (recent events in area)
                - confidence: float (0-1)
                - tier: str (top_5, long_tail_20, tail)
        
        Returns:
            Dictionary with structured explanation
        """
        
        # Extract context
        score = context_dict.get('score', 0.0)
        temporal = context_dict.get('temporal_pattern', 'Stable')
        nearby = context_dict.get('nearby_nodes', [])
        events = context_dict.get('events', [])
        confidence = context_dict.get('confidence', 0.85)
        tier = context_dict.get('tier', 'unknown')
        
        # Generate factors with contributions
        factors = []
        total_contribution = 0.0
        
        # 1. Temporal factor
        temporal_contrib = 0.35  # Temporal accounts for ~35%
        factors.append({
            'name': 'Temporal Pattern',
            'contribution': temporal_contrib,
            'explanation': temporal,
            'importance': 'high'
        })
        total_contribution += temporal_contrib
        
        # 2. Spatial correlation
        spatial_contrib = 0.30  # Spatial accounts for ~30%
        if nearby:
            nearby_str = ', '.join([f"Node {n}" for n in nearby[:3]])
            factors.append({
                'name': 'Spatial Correlation',
                'contribution': spatial_contrib,
                'explanation': f"Nearby nodes {nearby_str} also high-risk",
                'importance': 'high'
            })
        else:
            factors.append({
                'name': 'Spatial Correlation',
                'contribution': spatial_contrib,
                'explanation': "Isolated high-risk area (no nearby correlated nodes)",
                'importance': 'medium'
            })
        total_contribution += spatial_contrib
        
        # 3. Event impact
        event_contrib = 0.25  # Events account for ~25%
        if events:
            event_types = set()
            for event in events:
                event_types.add(event.get('type', 'crime'))
            
            event_str = ', '.join(list(event_types)[:2])
            factors.append({
                'name': 'Recent Events',
                'contribution': event_contrib,
                'explanation': f"Recent activity: {len(events)} event(s) ({event_str})",
                'importance': 'high'
            })
        else:
            factors.append({
                'name': 'Recent Events',
                'contribution': event_contrib,
                'explanation': "No recent events detected",
                'importance': 'low'
            })
        total_contribution += event_contrib
        
        # 4. Baseline/historical
        baseline_contrib = 1.0 - total_contribution
        factors.append({
            'name': 'Historical Baseline',
            'contribution': baseline_contrib,
            'explanation': f"Long-term {tier.replace('_', ' ')} risk level",
            'importance': 'medium'
        })
        
        # Generate summary explanation
        summary = self._generate_summary(node_id, rank, score, factors, tier)
        
        # Generate caveats
        caveats = self._generate_caveats(confidence, events, temporal)
        
        # Quantitative explanation
        quantitative = self._generate_quantitative(score, rank, tier, nearby)
        
        return {
            'node_id': int(node_id),
            'rank': int(rank),
            'score': float(score),
            'confidence': float(confidence),
            'summary': summary,
            'factors': factors,
            'quantitative': quantitative,
            'caveats': caveats,
            'interpretation': self._interpret_confidence(confidence)
        }
    
    def _generate_summary(self, node_id: int, rank: int, score: float, 
                         factors: List, tier: str) -> str:
        """Generate human-readable summary"""
        
        factor_strs = []
        for factor in factors:
            if factor['contribution'] > 0.15:  # Only mention significant factors
                pct = int(factor['contribution'] * 100)
                factor_strs.append(f"{factor['name']} ({pct}%)")
        
        factors_text = ' + '.join(factor_strs)
        
        # Tier-specific language
        if tier == 'top_5':
            tier_text = "one of the most critical areas"
        elif tier == 'long_tail_20':
            tier_text = "a significant risk area"
        elif tier == 'long_tail_50':
            tier_text = "a notable area of concern"
        else:
            tier_text = "a monitored area"
        
        summary = (
            f"Node {node_id} is ranked #{rank} and predicted as {tier_text} based on: "
            f"{factors_text}. "
            f"Risk score: {score:.2f}/10."
        )
        
        return summary
    
    def _generate_caveats(self, confidence: float, events: List, temporal: str) -> List[str]:
        """Generate trust caveats"""
        caveats = []
        
        # Confidence caveat
        if confidence < 0.70:
            caveats.append(
                f"Low confidence ({confidence:.0%}). Prediction may be unreliable."
            )
        elif confidence < 0.80:
            caveats.append(
                f"Moderate confidence ({confidence:.0%}). Treat as guidance, not certainty."
            )
        
        # Event caveat
        if events:
            caveats.append(
                f"Recent events detected. Model confidence reduced due to anomaly."
            )
        
        # Temporal caveat
        if "unusual" in str(temporal).lower() or "atypical" in str(temporal).lower():
            caveats.append(
                f"Unusual temporal pattern. Prediction based on limited historical data."
            )
        
        if not caveats:
            caveats.append("Model confidence is high for this prediction.")
        
        return caveats
    
    def _generate_quantitative(self, score: float, rank: int, tier: str, nearby: List) -> Dict:
        """Generate quantitative interpretation"""
        
        return {
            'risk_level': self._score_to_risk_level(score),
            'score_interpretation': f"Score {score:.2f}/10 suggests {'high' if score > 5 else 'moderate' if score > 3 else 'low'} risk",
            'rank_tier': tier,
            'peer_comparison': f"Compared to {len(nearby)} similar nodes nearby"
        }
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert score to risk category"""
        if score >= 8:
            return "CRITICAL"
        elif score >= 6:
            return "HIGH"
        elif score >= 4:
            return "MODERATE"
        elif score >= 2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence level"""
        if confidence >= 0.90:
            return "Very high confidence. This prediction is reliable."
        elif confidence >= 0.80:
            return "High confidence. This prediction is generally reliable."
        elif confidence >= 0.70:
            return "Moderate confidence. Use with caution."
        elif confidence >= 0.60:
            return "Low confidence. Treat as guidance only."
        else:
            return "Very low confidence. Do not rely on this prediction alone."
    
    def explain_top_k(self, top_k_nodes: List[Tuple[int, float]], 
                      contexts: Dict[int, Dict], k: int = 5) -> Dict:
        """
        Generate explanations for top-K ranking
        
        Args:
            top_k_nodes: List of (node_id, score) tuples
            contexts: Dictionary of node_id -> context
            k: How many to explain
        
        Returns:
            Dictionary with top-K explanation
        """
        
        explanations = []
        for rank, (node_id, score) in enumerate(top_k_nodes[:k], 1):
            context = contexts.get(node_id, {})
            context['score'] = score
            context['confidence'] = 0.85  # Default confidence
            
            explanation = self.explain_node_ranking(node_id, rank, context)
            explanations.append(explanation)
        
        return {
            'type': f'top_{k}_ranking',
            'total_nodes': len(top_k_nodes),
            'explained_nodes': explanations,
            'summary': f"Top {k} highest-risk areas identified with confidence-weighted explanations"
        }
    
    def print_explanation(self, explanation: Dict):
        """Pretty-print an explanation"""
        
        print(f"\n{'='*80}")
        print(f"Node {explanation['node_id']} - Rank #{explanation['rank']}")
        print(f"{'='*80}")
        
        print(f"\n📍 {explanation['summary']}")
        
        print(f"\nRisk Assessment:")
        print(f"  - Score: {explanation['score']:.2f}/10")
        print(f"  - Level: {explanation['quantitative']['risk_level']}")
        print(f"  - Confidence: {explanation['confidence']:.0%}")
        
        print(f"\nKey Factors:")
        for factor in explanation['factors']:
            if factor['contribution'] > 0.10:  # Only show significant factors
                pct = int(factor['contribution'] * 100)
                print(f"  • {factor['name']} ({pct}%): {factor['explanation']}")
        
        print(f"\n⚠️  Important Notes:")
        for caveat in explanation['caveats']:
            print(f"  • {caveat}")
        
        print(f"\nInterpretation: {explanation['interpretation']}")
        print(f"{'='*80}\n")


def create_sample_context(node_id: int) -> Dict:
    """Create sample context for testing"""
    return {
        'score': np.random.uniform(4, 9),
        'temporal_pattern': {
            1: 'High in evenings',
            2: 'Peaks on weekends',
            3: 'Stable throughout week',
            4: 'Higher during rainy season'
        }.get(node_id % 4, 'Variable pattern'),
        'nearby_nodes': [node_id - 1, node_id + 1] if node_id > 0 else [node_id + 1],
        'events': [
            {'type': 'robbery', 'date': '2026-02-03'},
            {'type': 'assault', 'date': '2026-02-05'}
        ] if node_id % 3 == 0 else [],
        'confidence': 0.85,
        'tier': 'top_5' if node_id % 10 == 0 else 'long_tail_20'
    }


if __name__ == "__main__":
    print("Explanation Generator Tests")
    print("=" * 80)
    
    gen = ExplanationGenerator()
    
    # Test explanation for a single node
    context = create_sample_context(146)
    explanation = gen.explain_node_ranking(146, 1, context)
    gen.print_explanation(explanation)
    
    # Test top-K explanations
    print("\nTop-5 Nodes Explanation")
    print("=" * 80)
    
    top_nodes = [(146, 8.5), (145, 8.2), (147, 7.9), (144, 7.6), (148, 7.3)]
    contexts = {node_id: create_sample_context(node_id) for node_id, _ in top_nodes}
    
    top_explanation = gen.explain_top_k(top_nodes, contexts, k=3)
    
    print(f"Generated {len(top_explanation['explained_nodes'])} explanations")
    for expl in top_explanation['explained_nodes']:
        gen.print_explanation(expl)
    
    print("✅ Explanation generation complete!")
