"""
Mock LLM para testes de Phase 2
Simula respostas LLM sem chamar API real
"""
import json
import random
from typing import Dict, List

# Mock events database
MOCK_EVENTS = {
    "event_001": {
        "text": "Roubo a mão armada em Aldeota perto de shopping. Três suspeitos. Polícia acionada.",
        "severity": "HIGH",
        "crime_types": ["robbery", "armed_assault"],
        "affected_nodes": [45, 46, 47],
        "police_response": "fast"
    },
    "event_002": {
        "text": "Furto de celular em rua pública no Bairro de Fátima. Vítima mais um suspeito.",
        "severity": "LOW",
        "crime_types": ["theft"],
        "affected_nodes": [120, 121],
        "police_response": "slow"
    },
    "event_003": {
        "text": "Tráfico de drogas suspeito em Messejana. Grande movimento de pessoas. Resgistrado múltiplos carros.",
        "severity": "HIGH",
        "crime_types": ["drug_trafficking", "suspicious_activity"],
        "affected_nodes": [200, 201, 202, 203],
        "police_response": "delayed"
    },
    "event_004": {
        "text": "Briga de rua em Mucuripe. Duas pessoas feridas. Ambulância chamada.",
        "severity": "MEDIUM",
        "crime_types": ["assault", "fight"],
        "affected_nodes": [150, 151],
        "police_response": "moderate"
    },
    "event_005": {
        "text": "Homicídio em Pirambu. Corpo encontrado em via pública. Polícia investiga.",
        "severity": "CRITICAL",
        "crime_types": ["homicide", "violence"],
        "affected_nodes": [100, 101, 102],
        "police_response": "immediate"
    }
}

# Crime taxonomy
CRIME_TAXONOMY = {
    "homicide": 1.0,
    "robbery": 0.8,
    "armed_assault": 0.75,
    "drug_trafficking": 0.7,
    "theft": 0.3,
    "assault": 0.6,
    "fight": 0.4,
    "suspicious_activity": 0.5,
    "violence": 0.85,
}

SEVERITY_LEVELS = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.2
}

POLICE_RESPONSE_TIMES = {
    "immediate": 0.95,
    "fast": 0.75,
    "moderate": 0.5,
    "slow": 0.25,
    "delayed": 0.1
}


class MockLLM:
    """Simula LLM para testes sem chamar API real"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        self.call_count = 0
        self.cache = {}
    
    def parse_event(self, event_text: str) -> Dict:
        """
        Parse event text e retorna estrutura JSON
        Mock: usa event pré-existente ou simula resposta
        """
        self.call_count += 1
        
        # Check cache
        if event_text in self.cache:
            return self.cache[event_text]
        
        # Search in mock database
        for event_id, event_data in MOCK_EVENTS.items():
            if event_text.lower() in event_data["text"].lower() or \
               event_data["text"].lower() in event_text.lower():
                result = {
                    "success": True,
                    "event_id": event_id,
                    "severity": event_data["severity"],
                    "crime_types": event_data["crime_types"],
                    "affected_nodes": event_data["affected_nodes"],
                    "police_response": event_data["police_response"],
                    "confidence": 0.95
                }
                self.cache[event_text] = result
                return result
        
        # Random mock response if not found
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        crime_types = list(CRIME_TAXONOMY.keys())
        police_responses = list(POLICE_RESPONSE_TIMES.keys())
        
        result = {
            "success": True,
            "event_id": f"mock_{self.call_count}",
            "severity": random.choice(severities),
            "crime_types": random.sample(crime_types, k=random.randint(1, 3)),
            "affected_nodes": list(range(random.randint(0, 319), 
                                        random.randint(5, 319), 10))[:random.randint(2, 4)],
            "police_response": random.choice(police_responses),
            "confidence": random.uniform(0.7, 1.0)
        }
        
        self.cache[event_text] = result
        return result
    
    def extract_severity_score(self, severity_text: str) -> float:
        """Convert severity text to score 0-1"""
        for key, score in SEVERITY_LEVELS.items():
            if key.lower() == severity_text.lower():
                return score
        return 0.5
    
    def extract_crime_importance(self, crime_types: List[str]) -> float:
        """Calculate average importance of crime types"""
        if not crime_types:
            return 0.5
        
        scores = [CRIME_TAXONOMY.get(ct, 0.5) for ct in crime_types]
        return sum(scores) / len(scores)
    
    def get_call_count(self) -> int:
        """Return number of LLM calls made"""
        return self.call_count
    
    def reset_cache(self):
        """Clear cache for new tests"""
        self.cache = {}
        self.call_count = 0


def create_event_features(event_parsed: Dict, num_nodes: int = 319) -> Dict:
    """
    Convert parsed event into feature vectors
    Returns dict with feature arrays for each node
    """
    llm = MockLLM()
    
    severity_score = llm.extract_severity_score(event_parsed["severity"])
    crime_importance = llm.extract_crime_importance(event_parsed["crime_types"])
    police_response_score = POLICE_RESPONSE_TIMES.get(
        event_parsed.get("police_response", "slow"), 0.5
    )
    
    # Base features for affected nodes
    affected_nodes = event_parsed.get("affected_nodes", [])
    base_multiplier = severity_score * crime_importance * police_response_score
    
    # Create feature array (one number per node)
    features = {}
    for node_id in range(num_nodes):
        if node_id in affected_nodes:
            # Affected node: high feature value
            features[node_id] = base_multiplier
        else:
            # Unaffected node: no feature contribution
            features[node_id] = 0.0
    
    return features


# Test mock LLM
if __name__ == "__main__":
    llm = MockLLM()
    
    print("=" * 70)
    print("TESTE: Mock LLM")
    print("=" * 70 + "\n")
    
    test_texts = [
        "Roubo em Aldeota",
        "Homicídio em Pirambu",
        "Evento aleatório não previamente testado"
    ]
    
    for text in test_texts:
        result = llm.parse_event(text)
        print(f"Texto: {text}")
        print(f"  Severity: {result['severity']}")
        print(f"  Crime Types: {result['crime_types']}")
        print(f"  Affected Nodes: {result['affected_nodes']}")
        print(f"  Police Response: {result['police_response']}")
        print(f"  Confidence: {result['confidence']:.2f}\n")
    
    print(f"Total LLM calls: {llm.get_call_count()}")
    print("✅ Mock LLM working correctly!")
