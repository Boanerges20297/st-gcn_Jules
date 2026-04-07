import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import re
import os

# Importar serviço de LLM
try:
    from src.llm_service import get_gemini_api_keys, _call_model_with_rotation
except ImportError:
    # Fallback para execução direta
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.llm_service import get_gemini_api_keys, _call_model_with_rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_CHANNEL_CATALOG = {
    0: {'key': 'cvli_count', 'label': 'CVLI diário', 'description': 'Contagem diária de CVLI no território.'},
    1: {'key': 'vehicle_crime_pressure', 'label': 'Pressão de crimes veiculares', 'description': 'Canal ponderado para crimes patrimoniais e de veículos.'},
    2: {'key': 'territorial_tension', 'label': 'Tensão territorial', 'description': 'Índice estrutural de tensão faccional do território.'},
    3: {'key': 'weekday_monday', 'label': 'Segunda-feira', 'description': 'Indicador sazonal de segunda-feira.'},
    4: {'key': 'weekday_tuesday', 'label': 'Terça-feira', 'description': 'Indicador sazonal de terça-feira.'},
    5: {'key': 'weekday_wednesday', 'label': 'Quarta-feira', 'description': 'Indicador sazonal de quarta-feira.'},
    6: {'key': 'weekday_thursday', 'label': 'Quinta-feira', 'description': 'Indicador sazonal de quinta-feira.'},
    7: {'key': 'weekday_friday_weighted', 'label': 'Sexta-feira reforçada', 'description': 'Indicador sazonal com reforço específico de sexta-feira.'},
    8: {'key': 'weekday_saturday', 'label': 'Sábado', 'description': 'Indicador sazonal de sábado.'},
    9: {'key': 'weekday_sunday', 'label': 'Domingo', 'description': 'Indicador sazonal de domingo.'},
    10: {'key': 'month_january', 'label': 'Janeiro', 'description': 'Indicador sazonal de janeiro.'},
    11: {'key': 'month_february', 'label': 'Fevereiro', 'description': 'Indicador sazonal de fevereiro.'},
    12: {'key': 'month_march', 'label': 'Março', 'description': 'Indicador sazonal de março.'},
    13: {'key': 'month_april', 'label': 'Abril', 'description': 'Indicador sazonal de abril.'},
    14: {'key': 'month_may', 'label': 'Maio', 'description': 'Indicador sazonal de maio.'},
    15: {'key': 'month_june', 'label': 'Junho', 'description': 'Indicador sazonal de junho.'},
    16: {'key': 'month_july', 'label': 'Julho', 'description': 'Indicador sazonal de julho.'},
    17: {'key': 'month_august', 'label': 'Agosto', 'description': 'Indicador sazonal de agosto.'},
    18: {'key': 'month_september', 'label': 'Setembro', 'description': 'Indicador sazonal de setembro.'},
    19: {'key': 'month_october', 'label': 'Outubro', 'description': 'Indicador sazonal de outubro.'},
    20: {'key': 'month_november', 'label': 'Novembro', 'description': 'Indicador sazonal de novembro.'},
    21: {'key': 'month_december', 'label': 'Dezembro', 'description': 'Indicador sazonal de dezembro.'},
    22: {'key': 'weekend_flag', 'label': 'Fim de semana', 'description': 'Indicador binário de fim de semana.'},
    23: {'key': 'channel_23_reserved', 'label': 'Canal reservado 23', 'description': 'Canal reservado sem uso analítico explícito na camada atual.'},
    24: {'key': 'rolling_cvli_7d', 'label': 'Rolling CVLI 7d', 'description': 'Soma móvel de 7 dias do canal de CVLI.'},
    25: {'key': 'channel_25_reserved', 'label': 'Canal reservado 25', 'description': 'Canal reservado sem uso analítico explícito na camada atual.'},
    26: {'key': 'channel_26_reserved', 'label': 'Canal reservado 26', 'description': 'Canal reservado sem uso analítico explícito na camada atual.'},
    27: {'key': 'intelligence_signal', 'label': 'Sinal de inteligência', 'description': 'Canal ponderado de ocorrências classificadas como inteligência.'},
    28: {'key': 'global_cvli_context', 'label': 'Contexto global de CVLI', 'description': 'Agregado global diário de CVLI sobre todos os nós da região.'},
    29: {'key': 'holiday_flag', 'label': 'Feriado nacional', 'description': 'Indicador binário de feriado nacional brasileiro.'},
    30: {'key': 'cvp_hot_day_flag', 'label': 'Dia quente CVP', 'description': 'Indicador binário de dias quentes da regra CVP.'},
    31: {'key': 'precipitation_mm', 'label': 'Precipitação', 'description': 'Precipitação diária em milímetros.'},
    32: {'key': 'significant_rain_flag', 'label': 'Chuva significativa', 'description': 'Indicador binário para precipitação acima de 5 mm.'},
    33: {'key': 'momentum_delta_7d', 'label': 'Momentum 7d', 'description': 'Diferença entre o CVLI dos 7 dias recentes e a janela anterior de 7 dias.'},
    34: {'key': 'momentum_delta_14d', 'label': 'Momentum 14d', 'description': 'Diferença entre o CVLI dos 14 dias recentes e a janela anterior de 14 dias.'},
    35: {'key': 'momentum_delta_30d', 'label': 'Momentum 30d', 'description': 'Diferença entre o CVLI dos 30 dias recentes e a janela anterior de 30 dias.'},
    36: {'key': 'cold_streak_inverse', 'label': 'Cold streak inverso', 'description': 'Sequência de dias sem CVLI, codificada como valor negativo truncado em 30 dias.'},
}

class ExplanationGenerator:
    """
    Gerador de Explicações Híbrido (LLM + Heurística de Elite).
    Utiliza Gemini 2.0 Flash para análise estratégica e métricas reais como fallback.
    """
    
    def __init__(self, model=None, data_manager=None):
        self.model = model
        self.data_manager = data_manager

    def explain_node_ranking(self, 
                            node_id: int, 
                            rank: int,
                            context_dict: Dict) -> Dict:
        """
        Gera uma explicação baseada em métricas técnicas e heurísticas de explicabilidade do modelo.
        """
        # Dados Base
        name = context_dict.get('name', f"Localidade {node_id}")
        score = context_dict.get('score', 0.0)
        confidence = context_dict.get('confidence', 0.85)
        tier = context_dict.get('tier', 'monitorada')
        
        # Retorna apenas a Lógica Elite (Métricas Reais e Explicabilidade Técnica)
        return self._generate_elite_fallback(node_id, name, rank, score, confidence, tier, context_dict)

    def build_academic_node_ranking(
        self,
        node_id: int,
        rank: int,
        context_dict: Dict,
    ) -> Dict:
        name = context_dict.get('name', f"Localidade {node_id}")
        score = context_dict.get('score', 0.0)
        confidence = context_dict.get('confidence', 0.85)
        tier = context_dict.get('tier', 'monitorada')
        return self._build_academic_payload(node_id, name, rank, score, confidence, tier, context_dict)

    def get_feature_channel_catalog(self) -> Dict[str, Dict]:
        return {
            str(index): dict(meta)
            for index, meta in FEATURE_CHANNEL_CATALOG.items()
        }

    def _top_slice_label(self, rank: int, total_nodes: int) -> str:
        if total_nodes <= 0:
            return "top do ranking"
        top_slice_pct = max(1.0, round((rank / total_nodes) * 100.0, 1))
        if float(top_slice_pct).is_integer():
            pct_text = str(int(top_slice_pct))
        else:
            pct_text = f"{top_slice_pct:.1f}".replace('.', ',')
        return f"top {pct_text}% do ranking"

    def _summary_reason(self, cvli_recent: int, cvli_prev: int, nearby_impact: List[str], events: List[Dict], confidence: float) -> str:
        if cvli_recent > cvli_prev and cvli_recent > 0:
            return "A posição é sustentada por aceleração recente de CVLI na janela observada."
        if nearby_impact:
            return "A posição é sustentada por pressão espacial de áreas vizinhas também críticas."
        if events:
            return "A posição é sustentada por eventos críticos recentes associados ao território."
        if confidence >= 0.85:
            return "Mesmo sem um gatilho agudo isolado, o modelo enxerga um padrão estrutural recorrente e consistente para este território."
        return "A criticidade decorre da combinação atual de score, posição relativa e sinais estruturais aprendidos pelo modelo."

    def _build_contextual_fallback_factors(
        self,
        name: str,
        rank: int,
        score: float,
        confidence: float,
        context: Dict,
    ) -> List[Dict]:
        total_nodes = int(context.get('total_nodes') or 0)
        score_pct = float(context.get('score_pct', score * 10.0))
        avg_score_pct = float(context.get('avg_score_pct', 0.0) or 0.0)
        temporal_pattern = str(context.get('temporal_pattern') or '').strip().lower()

        factors: List[Dict] = []

        ranking_fragments = []
        if total_nodes > 0:
            ranking_fragments.append(f"permanece no {self._top_slice_label(rank, total_nodes)}")
        ranking_fragments.append(f"score atual de {score:.2f}/10")
        if avg_score_pct > 0 and score_pct >= avg_score_pct:
            ranking_fragments.append(f"acima da média operacional de {avg_score_pct:.1f}%")

        factors.append({
            'name': 'Posicionamento Relativo',
            'explanation': "🧭 **Persistência no ranking**: " + ", ".join(ranking_fragments) + ".",
            'importance': 'medium' if rank > 10 else 'high'
        })

        signal_text = (
            f"🎯 **Sinal modelado consistente**: A confiança de {confidence * 100.0:.1f}% indica que a criticidade "
            "não está apoiada em um único evento isolado, mas em um padrão espacial e temporal recorrente aprendido pelo modelo."
        )
        factors.append({
            'name': 'Consistência do Modelo',
            'explanation': signal_text,
            'importance': 'high' if confidence >= 0.9 else 'medium'
        })

        if temporal_pattern == 'increasing':
            fallback_text = "📈 **Trajetória subjacente**: Mesmo com poucos gatilhos explícitos nesta leitura, a série local ainda aponta inclinação de alta frente ao histórico recente."
        else:
            fallback_text = (
                "🕰️ **Pressão estrutural**: A ausência de pico agudo na janela curta não elimina a criticidade; "
                "o modelo mantém o território elevado por recorrência histórica e posição relativa no ranking."
            )
        factors.append({
            'name': 'Leitura Estrutural',
            'explanation': fallback_text,
            'importance': 'low'
        })

        return factors

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _safe_ratio(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        try:
            denominator = float(denominator)
            if abs(denominator) < 1e-9:
                return default
            return float(numerator) / denominator
        except (TypeError, ValueError, ZeroDivisionError):
            return default

    def _pt_number(self, value: float, decimals: int = 1) -> str:
        try:
            return f"{float(value):.{decimals}f}".replace('.', ',')
        except (TypeError, ValueError):
            return f"{0.0:.{decimals}f}".replace('.', ',')

    def _as_list(self, value) -> List:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.85:
            return 'Alta'
        if confidence >= 0.70:
            return 'Moderada'
        if confidence >= 0.55:
            return 'Baixa'
        return 'Muito baixa'

    def _format_street_reference(self, streets) -> Tuple[str, int]:
        if isinstance(streets, list):
            clean = [str(item).strip() for item in streets if str(item).strip()]
            return ', '.join(clean[:4]) if clean else 'Sem logradouros críticos recentes', len(clean)
        text = str(streets or '').strip()
        if not text or text.lower().startswith('sem logradouros'):
            return 'Sem logradouros críticos recentes', 0
        return text, len([part for part in re.split(r',|;', text) if part.strip()])

    def _make_factor(
        self,
        key: str,
        name: str,
        value: str,
        explanation: str,
        importance: str,
        priority: int,
    ) -> Dict:
        return {
            'key': key,
            'name': name,
            'value': value,
            'explanation': explanation,
            'importance': importance,
            'priority': priority,
        }

    def _build_confidence_assessment(self, rank: int, score: float, fallback_confidence: float, context: Dict) -> Dict:
        top_slice_pct = self._safe_float(context.get('top_slice_pct'), 100.0)
        score_zscore = self._safe_float(context.get('score_zscore'))
        score_gap_pct = self._safe_float(context.get('score_gap_pct'))
        cvli_recent_7 = self._safe_int(context.get('cvli_recent_7'))
        cvli_prev_7 = self._safe_int(context.get('cvli_prev_7'))
        cvli_recent_14 = self._safe_int(context.get('cvli_count_recent'))
        cvli_prev_14 = self._safe_int(context.get('cvli_count_prev'))
        cvli_recent_30 = self._safe_int(context.get('cvli_recent_30'))
        geo_neighbor_count = self._safe_int(context.get('geo_neighbor_count'))
        conflict_neighbor_count = self._safe_int(context.get('conflict_neighbor_count'))
        high_risk_neighbor_count = self._safe_int(context.get('high_risk_neighbor_count'))
        neighbor_mean_score = self._safe_float(context.get('neighbor_mean_score'))
        critical_event_count = self._safe_int(context.get('critical_event_count'))
        suppression_event_count = self._safe_int(context.get('suppression_event_count'))
        event_types_count = self._safe_int(context.get('event_types_count'))
        conflict_intensity = self._safe_float(context.get('conflict_intensity'))
        suppression_intensity = self._safe_float(context.get('suppression_intensity'))
        recent_intel_14 = self._safe_float(context.get('intel_recent_14'))
        recent_vehicles_14 = self._safe_float(context.get('vehicles_recent_14'))
        critical_streets_count = self._safe_int(context.get('critical_streets_count'))
        tension_index = self._safe_float(context.get('tension_index'))
        faction = str(context.get('faction') or 'NEUTRO').upper()

        rank_strength = self._clamp((100.0 - top_slice_pct) / 100.0, 0.0, 1.0)
        score_separation = self._clamp((score_zscore + 1.5) / 3.0, 0.0, 1.0)

        if cvli_recent_7 > cvli_prev_7 and cvli_recent_7 > 0:
            temporal_signal = 1.0
        elif cvli_recent_14 > cvli_prev_14 and cvli_recent_14 > 0:
            temporal_signal = 0.85
        elif cvli_recent_14 > 0 or cvli_recent_30 > 0:
            temporal_signal = 0.60
        else:
            temporal_signal = 0.15 if score_gap_pct > 0 else 0.05

        if geo_neighbor_count > 0:
            spatial_signal = self._clamp(
                (high_risk_neighbor_count / max(1, geo_neighbor_count)) * 0.55 +
                (neighbor_mean_score / 100.0) * 0.45,
                0.0,
                1.0,
            )
        else:
            spatial_signal = 0.0

        event_signal = self._clamp(
            (critical_event_count * 0.40) +
            (event_types_count * 0.10) +
            (conflict_intensity * 0.20),
            0.0,
            1.0,
        )
        intelligence_signal = self._clamp(
            (recent_intel_14 / 4.0) * 0.55 +
            (critical_streets_count / 5.0) * 0.45,
            0.0,
            1.0,
        )
        structural_signal = self._clamp((tension_index * 0.75) + (0.25 if faction != 'NEUTRO' else 0.0), 0.0, 1.0)
        market_signal = self._clamp(recent_vehicles_14 / 6.0, 0.0, 1.0)

        corroboration_domains = [
            temporal_signal >= 0.60,
            spatial_signal >= 0.45,
            event_signal >= 0.45,
            intelligence_signal >= 0.35,
            structural_signal >= 0.45,
            market_signal >= 0.35,
        ]
        corroboration_count = sum(1 for item in corroboration_domains if item)

        positive_total = (
            0.16 * rank_strength +
            0.14 * score_separation +
            0.16 * temporal_signal +
            0.14 * spatial_signal +
            0.12 * event_signal +
            0.10 * intelligence_signal +
            0.10 * structural_signal +
            0.04 * market_signal +
            0.04 * self._clamp(corroboration_count / 4.0, 0.0, 1.0)
        )

        suppression_penalty = 0.05 * self._clamp(suppression_intensity / 3.0, 0.0, 1.0)
        sparse_penalty = 0.05 if corroboration_count <= 1 else 0.0
        ambiguity_penalty = 0.04 if critical_event_count == 0 and cvli_recent_14 == 0 and high_risk_neighbor_count == 0 else 0.0
        conflict_ambiguity_penalty = 0.02 if suppression_event_count > critical_event_count and suppression_event_count > 0 else 0.0

        evidence_confidence = self._clamp(
            0.38 + positive_total - suppression_penalty - sparse_penalty - ambiguity_penalty - conflict_ambiguity_penalty,
            0.35,
            0.96,
        )
        final_confidence = self._clamp((0.35 * fallback_confidence) + (0.65 * evidence_confidence), 0.35, 0.96)

        components = [
            {
                'name': 'Separação no ranking',
                'direction': 'positive',
                'score_pct': round(rank_strength * 100.0, 1),
                'text': f"Posição #{rank} e permanência no top {self._pt_number(top_slice_pct)}% mantêm o território na primeira faixa de atenção.",
            },
            {
                'name': 'Distância estatística da média',
                'direction': 'positive',
                'score_pct': round(score_separation * 100.0, 1),
                'text': f"Score atual opera {self._pt_number(score_gap_pct)} pontos percentuais acima da média, com z-score {self._pt_number(score_zscore, 2)}.",
            },
            {
                'name': 'Corroboração temporal',
                'direction': 'positive',
                'score_pct': round(temporal_signal * 100.0, 1),
                'text': f"CVLI 7d={cvli_recent_7} contra {cvli_prev_7} na janela anterior; CVLI 14d={cvli_recent_14} contra {cvli_prev_14}.",
            },
            {
                'name': 'Corroboração espacial',
                'direction': 'positive',
                'score_pct': round(spatial_signal * 100.0, 1),
                'text': f"{high_risk_neighbor_count} vizinho(s) crítico(s) entre {geo_neighbor_count} adjacentes, com média vizinha de {self._pt_number(neighbor_mean_score)}%.",
            },
            {
                'name': 'Eventos observados',
                'direction': 'positive',
                'score_pct': round(event_signal * 100.0, 1),
                'text': f"{critical_event_count} evento(s) críticos, {event_types_count} tipologia(s) e intensidade acumulada de conflito {self._pt_number(conflict_intensity)}.",
            },
            {
                'name': 'Inteligência e ruas críticas',
                'direction': 'positive',
                'score_pct': round(intelligence_signal * 100.0, 1),
                'text': f"Sinal de inteligência recente em {self._pt_number(recent_intel_14)} e {critical_streets_count} referência(s) territoriais críticas.",
            },
            {
                'name': 'Estrutura territorial',
                'direction': 'positive',
                'score_pct': round(structural_signal * 100.0, 1),
                'text': f"Tensão territorial {self._pt_number(tension_index, 2)} e facção predominante {faction} sustentam a pressão estrutural do território.",
            },
        ]

        if suppression_penalty > 0 or sparse_penalty > 0 or ambiguity_penalty > 0 or conflict_ambiguity_penalty > 0:
            components.append({
                'name': 'Fatores de cautela',
                'direction': 'negative',
                'score_pct': round((suppression_penalty + sparse_penalty + ambiguity_penalty + conflict_ambiguity_penalty) * 100.0, 1),
                'text': f"Supressão recente={suppression_event_count} e baixa convergência entre evidências recomendam cautela na decisão operacional.",
            })

        caveats = []
        if suppression_event_count > 0:
            caveats.append('Há ação policial de supressão recente; isso pode mitigar parte do risco no curtíssimo prazo.')
        if corroboration_count <= 1:
            caveats.append('A previsão está apoiada em poucos blocos independentes de evidência, exigindo leitura preventiva.')
        if critical_event_count == 0 and cvli_recent_14 == 0 and high_risk_neighbor_count == 0:
            caveats.append('Não há gatilho agudo explícito nesta janela; a justificativa depende mais de padrão estrutural do que de evento recente.')

        confidence_text = (
            f"A confiança operacional resulta da convergência entre ranking, separação estatística, sinais temporais, contexto espacial, eventos observados, "
            f"inteligência territorial e tensão faccional. Nesta leitura, {corroboration_count} bloco(s) independentes sustentam a decisão."
        )

        return {
            'confidence': final_confidence,
            'confidence_pct': round(final_confidence * 100.0, 1),
            'confidence_label': self._confidence_label(final_confidence),
            'components': components,
            'caveats': caveats,
            'corroboration_count': corroboration_count,
            'confidence_text': confidence_text,
            'evidence_confidence': evidence_confidence,
            'rank_strength': rank_strength,
            'score_separation': score_separation,
            'temporal_signal': temporal_signal,
            'spatial_signal': spatial_signal,
            'event_signal': event_signal,
            'intelligence_signal': intelligence_signal,
            'structural_signal': structural_signal,
            'market_signal': market_signal,
            'suppression_penalty': suppression_penalty,
            'sparse_penalty': sparse_penalty,
            'ambiguity_penalty': ambiguity_penalty,
            'conflict_ambiguity_penalty': conflict_ambiguity_penalty,
        }

    def _build_academic_payload(
        self,
        node_id: int,
        name: str,
        rank: int,
        score: float,
        fallback_confidence: float,
        tier: str,
        context: Dict,
    ) -> Dict:
        confidence_bundle = self._build_confidence_assessment(rank, score, fallback_confidence, context)
        total_nodes = max(1, self._safe_int(context.get('total_nodes'), 1))
        score_pct = self._safe_float(context.get('score_pct'), score * 10.0)
        avg_score_pct = self._safe_float(context.get('avg_score_pct'))
        median_score_pct = self._safe_float(context.get('median_score_pct'))
        score_gap_pct = self._safe_float(context.get('score_gap_pct'))
        score_zscore = self._safe_float(context.get('score_zscore'))
        top_slice_pct = self._safe_float(context.get('top_slice_pct'))
        region_type = str(context.get('region_type') or '').upper()
        faction = str(context.get('faction') or 'NEUTRO').upper()
        tension_index = self._safe_float(context.get('tension_index'))
        cvli_recent_7 = self._safe_int(context.get('cvli_recent_7'))
        cvli_prev_7 = self._safe_int(context.get('cvli_prev_7'))
        cvli_recent_14 = self._safe_int(context.get('cvli_count_recent'))
        cvli_prev_14 = self._safe_int(context.get('cvli_count_prev'))
        cvli_recent_30 = self._safe_int(context.get('cvli_recent_30'))
        rolling_cvli_7d = self._safe_float(context.get('rolling_cvli_7d'))
        vehicles_recent_14 = self._safe_float(context.get('vehicles_recent_14'))
        intel_recent_14 = self._safe_float(context.get('intel_recent_14'))
        global_cvli_latest = self._safe_float(context.get('global_cvli_latest'))
        geo_neighbor_count = self._safe_int(context.get('geo_neighbor_count'))
        conflict_neighbor_count = self._safe_int(context.get('conflict_neighbor_count'))
        high_risk_neighbor_count = self._safe_int(context.get('high_risk_neighbor_count'))
        neighbor_mean_score = self._safe_float(context.get('neighbor_mean_score'))
        neighbor_max_score = self._safe_float(context.get('neighbor_max_score'))
        event_count = self._safe_int(context.get('events_count_total'))
        critical_event_count = self._safe_int(context.get('critical_event_count'))
        suppression_event_count = self._safe_int(context.get('suppression_event_count'))
        event_types = self._as_list(context.get('event_types'))
        total_event_intensity = self._safe_float(context.get('total_event_intensity'))
        conflict_intensity = self._safe_float(context.get('conflict_intensity'))
        suppression_intensity = self._safe_float(context.get('suppression_intensity'))
        rain_acc_14 = self._safe_float(context.get('rain_acc_14'))
        rainy_days_14 = self._safe_int(context.get('rainy_days_14'))
        holiday_days_14 = self._safe_int(context.get('holiday_days_14'))
        hot_days_14 = self._safe_int(context.get('hot_days_14'))
        weekend_days_14 = self._safe_int(context.get('weekend_days_14'))
        nearby_names = self._as_list(context.get('nearby_impact_names'))
        critical_streets_text, critical_streets_count = self._format_street_reference(context.get('critical_streets'))
        feature_channels_latest = dict(context.get('feature_channels_latest') or {})
        feature_channels_sum_7d = dict(context.get('feature_channels_sum_7d') or {})
        feature_channels_sum_14d = dict(context.get('feature_channels_sum_14d') or {})
        feature_channels_sum_30d = dict(context.get('feature_channels_sum_30d') or {})

        delta_cvli_7 = cvli_recent_7 - cvli_prev_7
        delta_cvli_14 = cvli_recent_14 - cvli_prev_14

        derived_indices = {
            'rank_inverse_pct': round((1.0 - self._safe_ratio(rank - 1, total_nodes, 0.0)) * 100.0, 2),
            'score_gap_pct': round(score_gap_pct, 3),
            'score_zscore': round(score_zscore, 4),
            'top_slice_pct': round(top_slice_pct, 3),
            'cvli_delta_7d': delta_cvli_7,
            'cvli_delta_14d': delta_cvli_14,
            'cvli_growth_ratio_7d': round(self._safe_ratio(delta_cvli_7, max(1, cvli_prev_7), 0.0), 4),
            'cvli_growth_ratio_14d': round(self._safe_ratio(delta_cvli_14, max(1, cvli_prev_14), 0.0), 4),
            'high_risk_neighbor_share': round(self._safe_ratio(high_risk_neighbor_count, max(1, geo_neighbor_count), 0.0), 4),
            'conflict_neighbor_share': round(self._safe_ratio(conflict_neighbor_count, max(1, geo_neighbor_count), 0.0), 4),
            'critical_event_share': round(self._safe_ratio(critical_event_count, max(1, event_count), 0.0), 4),
            'suppression_event_share': round(self._safe_ratio(suppression_event_count, max(1, event_count), 0.0), 4),
            'conflict_to_suppression_intensity_ratio': round(self._safe_ratio(conflict_intensity, max(1e-6, suppression_intensity), 0.0), 4),
            'intel_to_vehicle_ratio': round(self._safe_ratio(intel_recent_14, max(1e-6, vehicles_recent_14), 0.0), 4),
            'rainy_day_share_14d': round(self._safe_ratio(rainy_days_14, 14.0, 0.0), 4),
            'holiday_share_14d': round(self._safe_ratio(holiday_days_14, 14.0, 0.0), 4),
            'weekend_share_14d': round(self._safe_ratio(weekend_days_14, 14.0, 0.0), 4),
        }

        confidence_analysis = {
            'confidence_pct': confidence_bundle['confidence_pct'],
            'confidence_label': confidence_bundle['confidence_label'],
            'confidence_text': confidence_bundle['confidence_text'],
            'evidence_confidence': round(confidence_bundle.get('evidence_confidence', 0.0), 4),
            'fallback_confidence_seed': round(self._safe_float(context.get('heuristic_confidence_seed', fallback_confidence)), 4),
            'corroboration_count': confidence_bundle.get('corroboration_count', 0),
            'components': confidence_bundle.get('components', []),
            'caveats': confidence_bundle.get('caveats', []),
            'signal_scores': {
                'rank_strength': round(confidence_bundle.get('rank_strength', 0.0), 4),
                'score_separation': round(confidence_bundle.get('score_separation', 0.0), 4),
                'temporal_signal': round(confidence_bundle.get('temporal_signal', 0.0), 4),
                'spatial_signal': round(confidence_bundle.get('spatial_signal', 0.0), 4),
                'event_signal': round(confidence_bundle.get('event_signal', 0.0), 4),
                'intelligence_signal': round(confidence_bundle.get('intelligence_signal', 0.0), 4),
                'structural_signal': round(confidence_bundle.get('structural_signal', 0.0), 4),
                'market_signal': round(confidence_bundle.get('market_signal', 0.0), 4),
            },
            'penalties': {
                'suppression_penalty': round(confidence_bundle.get('suppression_penalty', 0.0), 4),
                'sparse_penalty': round(confidence_bundle.get('sparse_penalty', 0.0), 4),
                'ambiguity_penalty': round(confidence_bundle.get('ambiguity_penalty', 0.0), 4),
                'conflict_ambiguity_penalty': round(confidence_bundle.get('conflict_ambiguity_penalty', 0.0), 4),
            },
        }

        academic_text = {
            'abstract_ptbr': (
                f"{name} ocupa a posição #{rank} entre {total_nodes} territórios, com score de {self._pt_number(score, 2)}/10 e confiança analítica de "
                f"{self._pt_number(confidence_bundle['confidence_pct'])}%. O caso combina pressão relativa elevada, estrutura territorial de {faction} e "
                f"sinais temporais, espaciais e exógenos em graus distintos."
            ),
            'operational_note_ptbr': self._build_structured_result(name, rank, score, tier, context, confidence_bundle).get('manager_guidance'),
            'methodological_note_ptbr': (
                'Arquivo acadêmico paralelo, não destinado ao frontend. Os campos abaixo preservam variáveis brutas, derivadas, componentes de confiança e canais de features '
                'para uso em pesquisa complementar e modelos auxiliares de distribuição de equipes.'
            ),
        }

        return {
            'node_id': node_id,
            'name': name,
            'source': 'academic_parallel',
            'identification': {
                'region_type': region_type,
                'faction': faction,
                'tier': tier,
            },
            'ranking_metrics': {
                'rank_global': rank,
                'total_nodes': total_nodes,
                'top_slice_pct': round(top_slice_pct, 3),
                'score_0_10': round(score, 4),
                'score_pct': round(score_pct, 4),
            },
            'score_distribution_metrics': {
                'mean_score_pct': round(avg_score_pct, 4),
                'median_score_pct': round(median_score_pct, 4),
                'score_gap_pct': round(score_gap_pct, 4),
                'score_zscore': round(score_zscore, 4),
            },
            'temporal_metrics': {
                'cvli_recent_7d': cvli_recent_7,
                'cvli_previous_7d': cvli_prev_7,
                'cvli_recent_14d': cvli_recent_14,
                'cvli_previous_14d': cvli_prev_14,
                'cvli_recent_30d': cvli_recent_30,
                'rolling_cvli_7d_latest': round(rolling_cvli_7d, 4),
            },
            'spatial_metrics': {
                'geo_neighbor_count': geo_neighbor_count,
                'conflict_neighbor_count': conflict_neighbor_count,
                'high_risk_neighbor_count': high_risk_neighbor_count,
                'neighbor_mean_score_pct': round(neighbor_mean_score, 4),
                'neighbor_max_score_pct': round(neighbor_max_score, 4),
                'nearby_impact_names': nearby_names,
            },
            'event_metrics': {
                'events_count_total': event_count,
                'critical_event_count': critical_event_count,
                'suppression_event_count': suppression_event_count,
                'event_types_count': len(event_types),
                'event_types': event_types,
                'total_event_intensity': round(total_event_intensity, 4),
                'conflict_intensity': round(conflict_intensity, 4),
                'suppression_intensity': round(suppression_intensity, 4),
            },
            'intelligence_and_market_metrics': {
                'intel_recent_14d': round(intel_recent_14, 4),
                'vehicles_recent_14d': round(vehicles_recent_14, 4),
                'critical_streets_count': critical_streets_count,
                'critical_streets_reference': critical_streets_text,
                'global_cvli_latest': round(global_cvli_latest, 4),
            },
            'environmental_and_calendar_metrics': {
                'rain_acc_14d_mm': round(rain_acc_14, 4),
                'rainy_days_14d': rainy_days_14,
                'holiday_days_14d': holiday_days_14,
                'hot_days_14d': hot_days_14,
                'weekend_days_14d': weekend_days_14,
            },
            'territorial_structure_metrics': {
                'tension_index': round(tension_index, 4),
                'region_type': region_type,
                'faction': faction,
            },
            'derived_indices': derived_indices,
            'confidence_analysis': confidence_analysis,
            'feature_channels': {
                'catalog': self.get_feature_channel_catalog(),
                'latest_timestep': feature_channels_latest,
                'sum_7d': feature_channels_sum_7d,
                'sum_14d': feature_channels_sum_14d,
                'sum_30d': feature_channels_sum_30d,
            },
            'academic_text': academic_text,
        }

    def _build_justification_variables(
        self,
        name: str,
        rank: int,
        score: float,
        context: Dict,
        confidence_bundle: Dict,
    ) -> List[Dict]:
        total_nodes = max(1, self._safe_int(context.get('total_nodes'), 1))
        top_slice_pct = self._safe_float(context.get('top_slice_pct'), (rank / total_nodes) * 100.0)
        score_pct = self._safe_float(context.get('score_pct'), score * 10.0)
        avg_score_pct = self._safe_float(context.get('avg_score_pct'))
        score_gap_pct = self._safe_float(context.get('score_gap_pct'))
        score_zscore = self._safe_float(context.get('score_zscore'))
        region_type = str(context.get('region_type') or '').upper()
        faction = str(context.get('faction') or 'NEUTRO').upper()
        tension_index = self._safe_float(context.get('tension_index'))
        cvli_recent_7 = self._safe_int(context.get('cvli_recent_7'))
        cvli_prev_7 = self._safe_int(context.get('cvli_prev_7'))
        cvli_recent_14 = self._safe_int(context.get('cvli_count_recent'))
        cvli_prev_14 = self._safe_int(context.get('cvli_count_prev'))
        cvli_recent_30 = self._safe_int(context.get('cvli_recent_30'))
        vehicles_recent_14 = self._safe_float(context.get('vehicles_recent_14'))
        intel_recent_14 = self._safe_float(context.get('intel_recent_14'))
        rolling_cvli_7d = self._safe_float(context.get('rolling_cvli_7d'))
        global_cvli_latest = self._safe_float(context.get('global_cvli_latest'))
        geo_neighbor_count = self._safe_int(context.get('geo_neighbor_count'))
        high_risk_neighbor_count = self._safe_int(context.get('high_risk_neighbor_count'))
        conflict_neighbor_count = self._safe_int(context.get('conflict_neighbor_count'))
        neighbor_mean_score = self._safe_float(context.get('neighbor_mean_score'))
        neighbor_max_score = self._safe_float(context.get('neighbor_max_score'))
        event_count = self._safe_int(context.get('events_count_total'))
        critical_event_count = self._safe_int(context.get('critical_event_count'))
        suppression_event_count = self._safe_int(context.get('suppression_event_count'))
        event_types = self._as_list(context.get('event_types'))
        conflict_intensity = self._safe_float(context.get('conflict_intensity'))
        suppression_intensity = self._safe_float(context.get('suppression_intensity'))
        recent_rain_acc = self._safe_float(context.get('rain_acc_14'))
        rainy_days = self._safe_int(context.get('rainy_days_14'))
        holiday_days = self._safe_int(context.get('holiday_days_14'))
        hot_days = self._safe_int(context.get('hot_days_14'))
        weekend_days = self._safe_int(context.get('weekend_days_14'))
        structured_streets, critical_streets_count = self._format_street_reference(context.get('critical_streets'))
        nearby_names = self._as_list(context.get('nearby_impact_names'))
        corroboration_count = self._safe_int(confidence_bundle.get('corroboration_count'))

        candidates = [
            self._make_factor('rank', 'Posição global', f"#{rank} de {total_nodes}", f"📌 **Posição global**: {name} ocupa a posição #{rank} entre {total_nodes} territórios monitorados.", 'high', 1),
            self._make_factor('top_slice', 'Faixa percentual do ranking', f"top {self._pt_number(top_slice_pct)}%", f"🧭 **Faixa percentual**: o território permanece no top {self._pt_number(top_slice_pct)}% do ranking e segue na primeira faixa de atenção.", 'high' if rank <= 10 else 'medium', 2),
            self._make_factor('score', 'Score de risco', f"{self._pt_number(score_pct)}%", f"🎯 **Score atual**: o risco consolidado está em {self._pt_number(score_pct)}%, equivalente a {self._pt_number(score, 2)}/10.", 'high', 3),
            self._make_factor('gap', 'Desvio sobre a média', f"{self._pt_number(score_gap_pct)} p.p.", f"📈 **Desvio sobre a média**: o território opera {self._pt_number(score_gap_pct)} ponto(s) percentual(is) acima da média operacional de {self._pt_number(avg_score_pct)}%.", 'high' if score_gap_pct > 20 else 'medium', 4),
            self._make_factor('faction', 'Facção predominante', faction, f"🏴 **Facção predominante**: a dinâmica territorial atual está associada a {faction}, com efeito direto na leitura operacional.", 'medium', 7),
            self._make_factor('tension', 'Tensão territorial', self._pt_number(tension_index, 2), f"🔥 **Tensão territorial**: índice estrutural em {self._pt_number(tension_index, 2)} numa escala de 0 a 1, indicando pressão de base no território.", 'high' if tension_index >= 0.5 else 'medium', 5),
            self._make_factor('cvli_7d', 'CVLI nos últimos 7 dias', str(cvli_recent_7), f"🩸 **CVLI 7 dias**: {cvli_recent_7} ocorrência(s) na janela mais curta, contra {cvli_prev_7} na janela imediatamente anterior.", 'high' if cvli_recent_7 > 0 else 'medium', 6),
            self._make_factor('cvli_14d', 'CVLI nos últimos 14 dias', str(cvli_recent_14), f"📆 **CVLI 14 dias**: {cvli_recent_14} ocorrência(s) acumuladas na janela tática principal, ante {cvli_prev_14} no período anterior comparável.", 'high' if cvli_recent_14 > 0 else 'medium', 8),
            self._make_factor('neighbors', 'Vizinhos críticos', str(high_risk_neighbor_count), f"🧲 **Vizinhos críticos**: {high_risk_neighbor_count} vizinho(s) estão em faixa elevada, com média de {self._pt_number(neighbor_mean_score)}% e máximo de {self._pt_number(neighbor_max_score)}%.", 'high' if high_risk_neighbor_count > 0 else 'medium', 9),
            self._make_factor('events', 'Eventos exógenos recentes', str(event_count), f"🚨 **Eventos exógenos**: {event_count} evento(s) recente(s), sendo {critical_event_count} crítico(s) e {suppression_event_count} de supressão qualificada.", 'high' if event_count > 0 else 'medium', 10),
            self._make_factor('conflict_intensity', 'Intensidade de conflito', self._pt_number(conflict_intensity, 2), f"💥 **Intensidade de conflito**: o somatório ponderado dos eventos críticos atingiu {self._pt_number(conflict_intensity, 2)}.", 'high' if conflict_intensity > 0 else 'medium', 11),
            self._make_factor('suppression_intensity', 'Intensidade de supressão', self._pt_number(suppression_intensity, 2), f"🛡️ **Intensidade de supressão**: ações mitigadoras somaram {self._pt_number(suppression_intensity, 2)} e pedem leitura de mitigação de curto prazo.", 'medium' if suppression_intensity > 0 else 'low', 12),
            self._make_factor('intel', 'Inteligência policial recente', self._pt_number(intel_recent_14, 1), f"🕵️ **Inteligência recente**: o canal de inteligência acumulou {self._pt_number(intel_recent_14, 1)} ponto(s) ponderado(s) na janela de 14 dias.", 'high' if intel_recent_14 > 0 else 'medium', 13),
            self._make_factor('streets', 'Logradouros críticos', str(critical_streets_count), f"🛣️ **Logradouros críticos**: {critical_streets_count} referência(s) territorial(is) destacada(s). Destaques: {structured_streets}.", 'high' if critical_streets_count > 0 else 'medium', 14),
            self._make_factor('confidence_support', 'Corroboração da confiança', str(corroboration_count), f"✅ **Corroboração**: a decisão está apoiada em {corroboration_count} bloco(s) independentes de evidência convergente.", 'high' if corroboration_count >= 3 else 'medium', 15),
            self._make_factor('event_types', 'Tipos de evento', ', '.join(event_types[:4]) if event_types else 'Sem tipologia recente', f"🧾 **Tipologia recente**: {', '.join(event_types[:4]) if event_types else 'sem tipologia classificada na janela atual'}.", 'medium' if event_types else 'low', 16),
            self._make_factor('conflict_neighbors', 'Conflito adjacente', str(conflict_neighbor_count), f"⚔️ **Conflito adjacente**: {conflict_neighbor_count} conexão(ões) em adjacência conflitiva ampliam risco de transbordamento territorial.", 'medium' if conflict_neighbor_count > 0 else 'low', 17),
            self._make_factor('rolling_cvli', 'Rolling CVLI 7d', self._pt_number(rolling_cvli_7d, 1), f"🔄 **Rolling CVLI 7d**: a série móvel encerra em {self._pt_number(rolling_cvli_7d, 1)}, útil para separar pico isolado de permanência.", 'medium' if rolling_cvli_7d > 0 else 'low', 18),
            self._make_factor('vehicles', 'Mercados correlatos', self._pt_number(vehicles_recent_14, 1), f"🚗 **Mercados correlatos**: o canal de crimes patrimoniais e veículos registra {self._pt_number(vehicles_recent_14, 1)} ponto(s) recentes, indicando pressão contextual.", 'medium' if vehicles_recent_14 > 0 else 'low', 19),
            self._make_factor('nearby_names', 'Vizinhos que sustentam a leitura', ', '.join(nearby_names[:3]) if nearby_names else 'Sem destaque espacial', f"📡 **Vizinhos que sustentam a leitura**: {', '.join(nearby_names[:3]) if nearby_names else 'não houve destaque espacial nominal nesta leitura'}.", 'medium' if nearby_names else 'low', 20),
            self._make_factor('region', 'Região analítica', region_type or 'N/D', f"🗺️ **Região analítica**: a comparação foi feita dentro do recorte {region_type or 'N/D'}.", 'low', 21),
            self._make_factor('cvli_30d', 'Acumulado CVLI 30 dias', str(cvli_recent_30), f"🗓️ **Acumulado de 30 dias**: {cvli_recent_30} ocorrência(s) ajudam a medir persistência do padrão.", 'low' if cvli_recent_30 <= 0 else 'medium', 22),
            self._make_factor('rain', 'Chuva recente', f"{self._pt_number(recent_rain_acc, 1)} mm", f"🌧️ **Clima recente**: acumulado de {self._pt_number(recent_rain_acc, 1)} mm e {rainy_days} dia(s) com chuva significativa na janela observada.", 'low', 23),
            self._make_factor('calendar', 'Calendário sensível', f"fds={weekend_days}, fer={holiday_days}, quentes={hot_days}", f"📅 **Calendário sensível**: {weekend_days} dia(s) de fim de semana, {holiday_days} feriado(s) e {hot_days} dia(s) quentes CVP no período recente.", 'low', 24),
            self._make_factor('global_cvli', 'Contexto global CVLI', self._pt_number(global_cvli_latest, 1), f"🌐 **Contexto global**: o nível global de CVLI encerra a janela em {self._pt_number(global_cvli_latest, 1)}.", 'low', 25),
            self._make_factor('geo_neighbors', 'Vizinhos geográficos', str(geo_neighbor_count), f"📍 **Vizinhos geográficos**: há {geo_neighbor_count} território(s) adjacente(s) na malha espacial imediata.", 'low', 26),
        ]

        selected = sorted(candidates, key=lambda item: item['priority'])[:15]
        selected = sorted(selected, key=lambda item: (0 if item['importance'] == 'high' else 1 if item['importance'] == 'medium' else 2, item['priority']))

        return [
            {
                'name': item['name'],
                'value': item['value'],
                'explanation': item['explanation'],
                'importance': item['importance'],
            }
            for item in selected
        ]

    def _build_structured_result(
        self,
        name: str,
        rank: int,
        score: float,
        tier: str,
        context: Dict,
        confidence_bundle: Dict,
    ) -> Dict:
        score_pct = self._safe_float(context.get('score_pct'), score * 10.0)
        cvli_recent_14 = self._safe_int(context.get('cvli_count_recent'))
        cvli_prev_14 = self._safe_int(context.get('cvli_count_prev'))
        high_risk_neighbor_count = self._safe_int(context.get('high_risk_neighbor_count'))
        critical_event_count = self._safe_int(context.get('critical_event_count'))
        suppression_event_count = self._safe_int(context.get('suppression_event_count'))
        score_gap_pct = self._safe_float(context.get('score_gap_pct'))
        faction = str(context.get('faction') or 'NEUTRO').upper()
        top_slice_pct = self._safe_float(context.get('top_slice_pct'))
        tier_map = {
            'top_5': 'prioridade máxima',
            'long_tail_20': 'alta prioridade tática',
            'long_tail_50': 'atenção moderada',
            'tail': 'monitoramento preventivo',
        }

        executive_summary = (
            f"{name} entra na posição #{rank}, no top {self._pt_number(top_slice_pct)}% do ranking, com score de {self._pt_number(score, 2)}/10 "
            f"({self._pt_number(score_pct)}%). No ciclo atual, isso coloca o território em faixa de {tier_map.get(tier, 'monitoramento')} e pede acompanhamento direto da operação."
        )

        technical_basis = (
            f"A decisão técnica combina distância do score em relação à média ({self._pt_number(score_gap_pct)} p.p.), "
            f"evolução temporal de CVLI ({cvli_recent_14} na janela atual contra {cvli_prev_14} na anterior), pressão espacial "
            f"de {high_risk_neighbor_count} vizinho(s) crítico(s), {critical_event_count} evento(s) crítico(s) recente(s) "
            f"e a estrutura territorial associada à facção {faction}."
        )

        confidence_basis = (
            confidence_bundle.get('confidence_text') or
            'A confiança foi estimada por convergência entre sinais independentes do território.'
        )

        manager_guidance = (
            f"Para a operação, a leitura mais segura é tratar {name} como território de {'resposta imediata' if rank <= 10 else 'monitoramento reforçado'}, "
            f"com checagem de patrulhamento, inteligência e reação local."
        )
        if suppression_event_count > 0:
            manager_guidance += (
                f" Há {suppression_event_count} ação(ões) de supressão no histórico recente, o que pode aliviar parte do risco no curtíssimo prazo, "
                "sem descaracterizar a pressão de base."
            )

        methodology_note = (
            'Esta explicação usa regras auditáveis sobre sinais reais do pipeline: ranking, séries de CVLI, adjacência territorial, tensão faccional, '
            'eventos exógenos, inteligência e contexto operacional. A confiança indica consistência entre evidências para apoio à decisão, e não probabilidade causal calibrada.'
        )

        return {
            'executive_summary': executive_summary,
            'technical_basis': technical_basis,
            'confidence_basis': confidence_basis,
            'manager_guidance': manager_guidance,
            'methodology_note': methodology_note,
        }

    def _get_llm_explanation(self, name: str, rank: int, score: float, context: Dict) -> Optional[Dict]:
        """Chama o Gemini para gerar uma análise tática."""
        keys = get_gemini_api_keys()
        if not keys or os.environ.get('DISABLE_GENAI_FOR_TESTS') == '1':
            return None

        # Preparar dados para o prompt
        cvli_recent = context.get('cvli_count_recent', 0)
        cvli_prev = context.get('cvli_count_prev', 0)
        events = context.get('events', [])
        nearby = context.get('nearby_impact_names', [])
        
        # Estruturar descrição de eventos
        events_desc = ""
        if events:
            events_desc = "\nEventos recentes detectados:\n" + "\n".join([f"- {e.get('natureza', 'Ocorrência')}: {e.get('descricao', '')}" for e in events[:3]])

        prompt = (
            f"Você é um analista sênior de inteligência policial no Ceará (CPRAIO/CIOPS).\n"
            f"Analise por que o bairro **{name}** está na posição **#{rank}** de risco no ranking preditivo.\n\n"
            f"DADOS TÉCNICOS:\n"
            f"- Score de Risco: {score:.1f}/10\n"
            f"- Crimes (CVLI) na janela atual: {cvli_recent} (Janela anterior: {cvli_prev})\n"
            f"- Vizinhos influenciadores: {', '.join(nearby) if nearby else 'Nenhum próximo'}\n"
            f"{events_desc}\n\n"
            f"REGRAS:\n"
            f"1. Seja direto, técnico e use tom de comando.\n"
            f"2. Explique a influência do aumento de crimes ou dos vizinhos se houver.\n"
            f"3. Se houver ações policiais qualificadas de supressão, mencione mitigação parcial e temporária do risco.\n"
            f"4. Retorne APENAS um JSON com as chaves: 'summary' (resumo de 1 frase), 'factors' (lista de 2-3 strings com ícones), 'interpretation' (nota final de confiança).\n"
            f"5. Responda em Português do Brasil."
        )

        try:
            out = _call_model_with_rotation(prompt, keys)
            # Limpar saída para extrair JSON
            json_str = re.search(r'(\{.*\})', out, re.DOTALL).group(1)
            import json
            data = json.loads(json_str)
            
            # Formatar no padrão do frontend
            factors = []
            for f in data.get('factors', []):
                factors.append({
                    'name': 'Inteligência LLM',
                    'explanation': f,
                    'importance': 'high' if rank <= 10 else 'medium'
                })
            
            return {
                'node_id': context.get('node_id'),
                'name': name,
                'rank': rank,
                'score': score,
                'summary': data.get('summary', f"Análise estratégica para {name}"),
                'factors': factors,
                'interpretation': data.get('interpretation', "Baseado em padrões consistentes de inteligência."),
                'confidence': context.get('confidence', 0.9)
            }
        except Exception as e:
            logger.warning(f"Falha ao gerar explicação via Gemini: {e}")
            return None

    def _generate_elite_fallback(self, node_id, name, rank, score, confidence, tier, context) -> Dict:
        """Lógica de métricas oficiais do modelo (Sua 'Versão Elite')."""
        cvli_recent = context.get('cvli_count_recent', 0)
        cvli_prev = context.get('cvli_count_prev', 0)
        events = context.get('events', [])
        nearby_impact = context.get('nearby_impact_names', [])
        confidence_bundle = self._build_confidence_assessment(rank, score, confidence, context)
        final_confidence = confidence_bundle['confidence']
        factors = self._build_justification_variables(name, rank, score, context, confidence_bundle)

        tier_map = {'top_5': 'uma das 5 áreas mais críticas', 'long_tail_20': 'área de alta prioridade tática', 'long_tail_50': 'área de atenção moderada'}
        summary = (
            f"**{name}** está na posição **#{rank}** do ranking, com score **{score:.2f}/10** "
            f"e confiança **{final_confidence * 100.0:.1f}%** — identificada como {tier_map.get(tier, 'área monitorada')}. "
            f"{self._summary_reason(cvli_recent, cvli_prev, nearby_impact, events, final_confidence)}"
        )

        structured_result = self._build_structured_result(name, rank, score, tier, context, confidence_bundle)

        return {
            'node_id': node_id,
            'name': name,
            'rank': rank,
            'score': score,
            'confidence': final_confidence,
            'confidence_pct': confidence_bundle['confidence_pct'],
            'confidence_label': confidence_bundle['confidence_label'],
            'confidence_components': confidence_bundle['components'],
            'confidence_caveats': confidence_bundle['caveats'],
            'confidence_explanation': confidence_bundle['confidence_text'],
            'summary': summary,
            'structured_result': structured_result,
            'factors': factors,
            'interpretation': self._interpret_confidence(final_confidence, confidence_bundle),
            'source': 'elite_contextual'
        }

    def _interpret_confidence(self, confidence: float, confidence_bundle: Optional[Dict] = None) -> str:
        corroboration_count = self._safe_int((confidence_bundle or {}).get('corroboration_count'))
        if confidence >= 0.90:
            return f"Confiança **alta** para decisão operacional, com {corroboration_count} bloco(s) independentes sustentando a leitura."
        if confidence >= 0.75:
            return f"Confiança **moderada para alta**. Há convergência suficiente entre sinais temporais, espaciais e estruturais ({corroboration_count} bloco(s) de evidência)."
        if confidence >= 0.60:
            return f"Confiança **moderada**. A previsão serve para prevenção, mas ainda depende de validação em campo por haver apenas {corroboration_count} bloco(s) fortes de evidência."
        return "Confiança **baixa**. Use como orientação preventiva e não como evidência conclusiva isolada."
