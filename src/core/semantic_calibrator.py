"""
Semantic Calibrator — Detecção de padrões similares no histórico de calibrações

Usa embeddings para encontrar calibrações passadas similares e recomendar
deltas baseado no que funcionou antes. Sem custo, sem latência (embeddings locais).
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("[Semantic] sentence_transformers não instalado. Usando heurísticas apenas.")


class SemanticCalibrator:
    """
    Analisa histórico de calibrações passadas usando embeddings semânticos.
    Encontra padrões similares e recomenda deltas baseado no sucesso anterior.
    """

    def __init__(self, base_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            base_dir: Diretório raiz do projeto
            model_name: Modelo de embeddings (pequeno e rápido por padrão)
        """
        self.base_dir = base_dir
        self.model_name = model_name
        self.calibration_state_path = os.path.join(base_dir, 'data', 'calibration_state.json')
        
        # Carregar embeddings model se disponível
        self.model = None
        self.embeddings_cache = {}  # pattern_hash -> embedding
        
        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"[Semantic] ✅ Modelo '{model_name}' carregado")
            except Exception as e:
                logger.warning(f"[Semantic] ⚠️ Erro ao carregar modelo: {e}")
        
        # Histórico processado
        self.calibration_history = []
        self._load_calibration_history()

    def _load_calibration_history(self):
        """Carrega histórico de calibrações do arquivo de estado."""
        try:
            if os.path.exists(self.calibration_state_path):
                with open(self.calibration_state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # Extrair histórico de cada região
                    for region, reg_data in state.items():
                        history = reg_data.get('history', [])
                        for entry in history:
                            self.calibration_history.append({
                                'region': region,
                                'timestamp': entry.get('timestamp'),
                                'trigger': entry.get('trigger'),
                                'metric': self._extract_metric_from_trigger(entry.get('trigger')),
                                'old_params': entry.get('old_params', {}),
                                'new_params': entry.get('new_params', {}),
                                'step': entry.get('step'),
                                'event': entry.get('event'),  # 'full_rollback' etc
                            })
                    
                    logger.info(f"[Semantic] ✅ {len(self.calibration_history)} históricos carregados")
        except Exception as e:
            logger.warning(f"[Semantic] ⚠️ Erro ao carregar histórico: {e}")

    def _extract_metric_from_trigger(self, trigger: str) -> Optional[str]:
        """Extrai métrica (p20, p10, faction_coverage) do trigger string."""
        if not trigger:
            return None
        
        if 'p20' in trigger.lower():
            return 'p20'
        elif 'p10' in trigger.lower():
            return 'p10'
        elif 'faction_coverage' in trigger.lower() or 'faction' in trigger.lower():
            return 'faction_coverage'
        
        return None

    def _create_pattern_description(self, region: str, metric: str, 
                                    current_value: float, threshold: float) -> str:
        """
        Cria descrição semântica do padrão de degradação.
        Exemplo: "fortaleza p20 degraded from 0.70 to 0.68 metric"
        """
        severity = "critical" if current_value < 0.50 else "high" if current_value < 0.65 else "medium"
        deficit = (threshold - current_value) / max(threshold, 0.01)
        
        pattern = (
            f"{region} {metric} {severity} degradation "
            f"current_value {current_value:.2f} threshold {threshold:.2f} "
            f"deficit {deficit:.2f}"
        )
        return pattern

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Obtém embedding do texto, com cache."""
        if not self.model:
            return None
        
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"[Semantic] ⚠️ Erro ao criar embedding: {e}")
            return None

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridade cosseno entre dois embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def find_similar_patterns(self, region: str, metric: str, 
                             current_value: float, threshold: float,
                             top_k: int = 3) -> List[Dict]:
        """
        Encontra calibrações similares no histórico.
        
        Returns:
            Lista de até top_k calibrações similares com score de similaridade
        """
        if not self.model or not self.calibration_history:
            return []
        
        # Criar padrão atual
        current_pattern = self._create_pattern_description(region, metric, current_value, threshold)
        current_embedding = self._get_embedding(current_pattern)
        
        if current_embedding is None:
            return []
        
        # Buscar padrões similares
        similarities = []
        
        for hist_entry in self.calibration_history:
            # Filtrar por região e métrica
            if hist_entry['region'] != region or hist_entry['metric'] != metric:
                continue
            
            # Pular rollbacks (nos queremos successos anteriores)
            if hist_entry.get('event') == 'full_rollback':
                continue
            
            hist_pattern = self._create_pattern_description(
                hist_entry['region'],
                hist_entry['metric'],
                self._extract_old_value_from_params(hist_entry['old_params']),
                threshold
            )
            
            hist_embedding = self._get_embedding(hist_pattern)
            if hist_embedding is None:
                continue
            
            similarity = self._cosine_similarity(current_embedding, hist_embedding)
            similarities.append({
                'similarity': similarity,
                'timestamp': hist_entry['timestamp'],
                'old_params': hist_entry['old_params'],
                'new_params': hist_entry['new_params'],
                'metric': metric,
                'region': region,
            })
        
        # Retornar top-k mais similares
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def _extract_old_value_from_params(self, params: Dict) -> float:
        """Extrai valor antigo (aproximação) dos parâmetros."""
        # Usar tension_factor como proxy de degradação
        return params.get('tension_factor', 0.80)

    def compute_optimized_delta(self, region: str, metric: str,
                                current_value: float, threshold: float,
                                base_delta: float) -> Tuple[float, Dict]:
        """
        Calcula delta otimizado baseado em padrões similares.
        
        Returns:
            (adjusted_delta, explanation_dict)
        """
        similar_patterns = self.find_similar_patterns(region, metric, current_value, threshold)
        
        if not similar_patterns:
            # Nenhum padrão similar — usar heurística padrão
            return base_delta, {
                'source': 'heuristic',
                'reason': 'No similar patterns found in history',
                'similarity_score': 0.0,
            }
        
        # Usar pattern mais similar
        best_match = similar_patterns[0]
        similarity_score = best_match['similarity']
        
        # Se muito similar (>0.85), usar exatamente o que funcionou antes
        if similarity_score > 0.85:
            # Extrair deltas que foram aplicados com sucesso
            old_bias = best_match['old_params'].get('tag_bias_direct', 2.00)
            new_bias = best_match['new_params'].get('tag_bias_direct', 2.00)
            successful_delta = new_bias - old_bias
            
            return successful_delta, {
                'source': 'semantic_exact_match',
                'reason': f'Pattern very similar (similarity={similarity_score:.3f}). Using exact delta from {best_match["timestamp"]}',
                'similarity_score': similarity_score,
                'previous_timestamp': best_match['timestamp'],
            }
        
        # Se parcialmente similar (0.60-0.85), usar como feedback
        elif similarity_score > 0.60:
            # Suavizar o ajuste baseado em confiança da similaridade
            adjusted_delta = base_delta * (0.8 + similarity_score * 0.2)
            
            return adjusted_delta, {
                'source': 'semantic_partial_match',
                'reason': f'Partial pattern match (similarity={similarity_score:.3f}). Scaling delta by {(0.8 + similarity_score * 0.2):.2f}',
                'similarity_score': similarity_score,
                'previous_timestamp': best_match['timestamp'],
            }
        
        # Pouco similar — usar heurística com feedback
        else:
            return base_delta, {
                'source': 'heuristic_with_feedback',
                'reason': f'Weak similarity ({similarity_score:.3f}). Using standard heuristic',
                'similarity_score': similarity_score,
            }

    def get_statistics(self) -> Dict:
        """Retorna estatísticas do histórico semântico."""
        return {
            'total_entries': len(self.calibration_history),
            'regions': list(set(h['region'] for h in self.calibration_history)),
            'metrics': list(set(h['metric'] for h in self.calibration_history if h['metric'])),
            'model_loaded': self.model is not None,
            'model_name': self.model_name if self.model else None,
        }
