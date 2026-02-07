"""
Event Anomaly Detector
Parses exogenous events and detects anomalies

Author: ST-GCN Enhanced System
Date: Feb 2026

Features:
- Heuristic-based event parsing (MVP)
- Severity classification (0-1)
- Anomaly detection
- Confidence reduction calculation
"""

import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventAnomalyDetector:
    """
    Parse event text and detect anomalies
    
    MVP approach: Keyword-based severity scoring
    Future: Upgrade to real LLM (Google Gemini, OpenAI, etc.)
    """
    
    def __init__(self):
        """Initialize with calibrated keyword severity mapping"""
        
        # Keyword -> Severity scores (0-1)
        self.severity_keywords = {
            # Critical severity (1.0)
            'homicídio|morte|corpo|cadáver': 1.0,
            'massacre|ataque em grupo': 1.0,
            
            # High severity (0.8)
            'roubo|assalto|armado|bala|tiroteio': 0.8,
            'sequestro|rapto|cárcere': 0.8,
            'explosão|bomba': 0.8,
            
            # Medium-high severity (0.7)
            'tráfico|droga|cocaína|maconha|crack': 0.7,
            'roubo de carro|veículo|auto': 0.7,
            'execução|morte suspeita': 0.7,
            
            # Medium severity (0.5)
            'briga|agressão|espancamento': 0.5,
            'vandalismo|incêndio': 0.5,
            'furto|roubo leve': 0.5,
            'invasão|arrombamento': 0.5,
            
            # Low severity (0.3)
            'furto|roubo de objeto': 0.3,
            'dano patrimonial': 0.3,
            'desentendimento|discussão': 0.3
        }
        
        # Keywords that reduce severity (modifiers)
        self.mitigating_keywords = {
            'tentativa': 0.8,  # Reduce by 20%
            'suspeita': 0.9,   # Reduce by 10%
            'relato': 0.95,    # Reduce by 5%
        }
        
        # Location patterns (rough scale for area coverage)
        self.location_multiplier = {
            'centro': 1.2,      # High traffic area
            'aldeota|barro': 1.1,
            'meireles|praia': 1.1,
            'canindezinho': 0.9,
            'mucuripe': 0.8,
        }
    
    def parse_event(self, event_text: str) -> Dict:
        """
        Parse a single event and classify severity
        
        Args:
            event_text: Event description (string)
        
        Returns:
            Dictionary with:
            - severity: float [0-1]
            - crime_types: list of identified crime types
            - anomaly_flag: bool (True if severity > 0.6)
            - confidence_reduction: float (amount to reduce model confidence)
        """
        
        if not event_text or len(event_text.strip()) == 0:
            return {
                'severity': 0.0,
                'crime_types': [],
                'anomaly_flag': False,
                'confidence_reduction': 0.0,
                'raw_text': event_text
            }
        
        # Convert to lowercase for matching
        text_lower = event_text.lower()
        
        # Find matching severity
        severity = 0.0
        matched_types = []
        
        for keywords, base_severity in self.severity_keywords.items():
            keyword_list = keywords.split('|')
            for kw in keyword_list:
                # Simple regex match
                if re.search(r'\b' + kw + r'\b', text_lower):
                    if base_severity > severity:
                        severity = base_severity
                    if kw not in matched_types:
                        matched_types.append(kw)
                    break
        
        # Apply mitigating factors
        for mitigation_kw, reduction_factor in self.mitigating_keywords.items():
            if re.search(r'\b' + mitigation_kw + r'\b', text_lower):
                severity = severity * reduction_factor
        
        # Apply location multiplier
        for location_kw, multiplier in self.location_multiplier.items():
            if re.search(r'\b' + location_kw + r'\b', text_lower):
                severity = severity * multiplier
                break
        
        # Clip to [0, 1]
        severity = min(1.0, max(0.0, severity))
        
        # Determine anomaly flag
        anomaly_flag = severity > 0.6
        
        # Calculate confidence reduction
        # Higher severity = more confidence reduction
        # Max 30% reduction (confidence * 0.7)
        confidence_reduction = severity * 0.3
        
        return {
            'severity': float(severity),
            'crime_types': matched_types,
            'anomaly_flag': bool(anomaly_flag),
            'confidence_reduction': float(confidence_reduction),
            'raw_text': event_text
        }
    
    def parse_events_batch(self, events_list: List[str]) -> List[Dict]:
        """
        Parse multiple events efficiently
        
        Args:
            events_list: List of event descriptions
        
        Returns:
            List of parsed events
        """
        return [self.parse_event(event) for event in events_list]
    
    def aggregate_event_severity(self, parsed_events: List[Dict]) -> Dict:
        """
        Aggregate severity from multiple events
        
        Takes the maximum severity across all events
        (one critical event makes whole day high-risk)
        
        Args:
            parsed_events: List of parsed event results
        
        Returns:
            Aggregated dictionary
        """
        
        if not parsed_events:
            return {
                'max_severity': 0.0,
                'anomaly_flag': False,
                'confidence_reduction': 0.0,
                'event_count': 0,
                'critical_events': []
            }
        
        severities = [e['severity'] for e in parsed_events]
        max_severity = max(severities)
        
        # Find critical events
        critical = [e for e in parsed_events if e['severity'] > 0.7]
        
        return {
            'max_severity': float(max_severity),
            'confidence_reduction': float(max_severity * 0.3),
            'anomaly_flag': max_severity > 0.6,
            'event_count': len(parsed_events),
            'critical_event_count': len(critical),
            'average_severity': float(sum(severities) / len(severities)),
            'critical_events': critical
        }
    
    def explain_severity(self, parsed_event: Dict) -> str:
        """
        Generate human-readable explanation for severity classification
        
        Args:
            parsed_event: Result from parse_event()
        
        Returns:
            Explanation string
        """
        
        severity = parsed_event['severity']
        
        if severity == 0:
            return "No anomaly detected in event text."
        elif severity < 0.3:
            return f"Low severity event ({severity:.1%}). Minimal impact on model confidence."
        elif severity < 0.6:
            return f"Medium severity event ({severity:.1%}). Model slightly less confident."
        elif severity < 0.8:
            return f"High severity event ({severity:.1%}). Model significantly less confident."
        else:
            return f"Critical severity event ({severity:.1%}). Model confidence substantially reduced."


def load_default_detector() -> EventAnomalyDetector:
    """Factory function to get default detector"""
    return EventAnomalyDetector()


# Example usage
if __name__ == "__main__":
    detector = EventAnomalyDetector()
    
    # Test cases
    test_events = [
        "Homicídio em Aldeota",
        "Tentativa de roubo em Meireles",
        "Relato de furto de bicicleta no Centro",
        "Tiroteio no Barro, suspeita de tráfico",
        "Discussão entre vizinhos",
        "Explosão suspeita em Canindezinho",
    ]
    
    print("Event Anomaly Detection Tests")
    print("=" * 80)
    
    for event_text in test_events:
        result = detector.parse_event(event_text)
        explanation = detector.explain_severity(result)
        
        print(f"\nEvent: {event_text}")
        print(f"  Severity: {result['severity']:.3f}")
        print(f"  Anomaly: {result['anomaly_flag']}")
        print(f"  Crime types: {result['crime_types']}")
        print(f"  Confidence reduction: {result['confidence_reduction']:.1%}")
        print(f"  → {explanation}")
    
    print("\n" + "=" * 80)
    
    # Test batch processing
    print("\nBatch Processing Test")
    batch_results = detector.parse_events_batch(test_events)
    aggregated = detector.aggregate_event_severity(batch_results)
    
    print(f"  Events processed: {aggregated['event_count']}")
    print(f"  Max severity: {aggregated['max_severity']:.3f}")
    print(f"  Critical events: {aggregated['critical_event_count']}")
    print(f"  Overall anomaly flag: {aggregated['anomaly_flag']}")
    print(f"  Model confidence reduction: {aggregated['confidence_reduction']:.1%}")
