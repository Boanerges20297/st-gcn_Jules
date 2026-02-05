#!/usr/bin/env python
"""
validate_stgcn_with_ranking.py
Helper function to validate ST-GCN predictions with ranking model in real-time
"""

import numpy as np
from typing import Tuple, Optional


def validate_and_reorder_predictions(stgcn_scores: np.ndarray,
                                     ranking_validator,
                                     node_features: Optional[np.ndarray] = None,
                                     use_validator: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Validate ST-GCN predictions using ranking model
    
    Args:
        stgcn_scores: (N,) ST-GCN raw scores
        ranking_validator: RankingInference instance (can be None)
        node_features: (N, D) Optional node features for ranking
        use_validator: Whether to use validator if available
    
    Returns:
        validated_scores: (N,) Final scores after validation
        metadata: dict with validation info
    """
    
    metadata = {
        'validation_used': False,
        'concordance': 1.0,
        'score_source': 'stgcn_only'
    }
    
    # No ranking validator or features
    if not use_validator or ranking_validator is None or node_features is None:
        return stgcn_scores, metadata
    
    # Run validation
    try:
        combined_scores, top_indices = ranking_validator.validate_stgcn_predictions(
            stgcn_scores, node_features, top_k=5
        )
        
        # Get concordance
        stgcn_top5 = np.argsort(-stgcn_scores)[:5]
        validated_top5 = top_indices
        overlap = len(set(stgcn_top5) & set(validated_top5))
        concordance = overlap / 5.0
        
        metadata['validation_used'] = True
        metadata['concordance'] = concordance
        metadata['score_source'] = f'st-gcn(70%) + ranking(30%)'
        
        return combined_scores, metadata
    
    except Exception as e:
        print(f"[WARNING] Ranking validation failed: {e}")
        return stgcn_scores, metadata
