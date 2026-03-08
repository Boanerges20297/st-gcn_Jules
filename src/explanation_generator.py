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
            f"3. Se houver ações policiais (supressão), mencione que o risco está sendo mitigado.\n"
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
        
        factors = []
        
        if cvli_recent > 0:
            diff = cvli_recent - cvli_prev
            if diff > 0:
                trend_text = f"📈 **Aumento real de criminalidade**: {cvli_recent} ocorrências registradas recentemente (alta de {diff} em relação ao período anterior)."
                importance = 'high'
            elif cvli_recent > 2:
                trend_text = f"📊 **Persistência de crimes**: Manutenção de {cvli_recent} ocorrências na janela atual, indicando inércia criminal."
                importance = 'medium'
            else:
                trend_text = f"🔄 **Base histórica**: Histórico de {cvli_recent} ocorrência(s) em área de monitoramento constante."
                importance = 'low'
            factors.append({'name': 'Padrão Temporal', 'explanation': trend_text, 'importance': importance})

        if nearby_impact:
            factors.append({
                'name': 'Correlação Espacial',
                'explanation': f"📍 **Efeito de contágio**: Influência direta do risco elevado nos vizinhos: {', '.join(nearby_impact[:3])}.",
                'importance': 'high'
            })

        if events:
            criticos = [e for e in events if e.get('is_suppression') is False]
            supressoes = [e for e in events if e.get('is_suppression') is True]
            if criticos:
                factors.append({'name': 'Eventos Críticos', 'explanation': f"⚡ **Alerta de Conflito**: {len(criticos)} evento(s) grave(s) (homicídios/ataques) registrados no setor recentemente.", 'importance': 'high'})
            if supressoes:
                factors.append({'name': 'Ações de Supressão', 'explanation': f"🛡️ **Presença Policial**: {len(supressoes)} ação(ões) de supressão detectada(s). O risco está sendo mitigado ativamente.", 'importance': 'medium'})

        if not factors:
            factors.append({'name': 'Histórico Estático', 'explanation': "🔄 **Área de Atenção Permanente**: O risco é derivado da base estatística de longo prazo e densidade populacional.", 'importance': 'low'})

        tier_map = {'top_5': 'uma das 5 áreas mais críticas', 'long_tail_20': 'área de alta prioridade tática', 'long_tail_50': 'área de atenção moderada'}
        summary = f"**{name}** está na posição **#{rank}** do ranking — identificada como {tier_map.get(tier, 'área monitorada')}."

        return {
            'node_id': node_id, 'name': name, 'rank': rank, 'score': score, 'confidence': confidence,
            'summary': summary, 'factors': factors, 'interpretation': self._interpret_confidence(confidence)
        }

    def _interpret_confidence(self, confidence: float) -> str:
        if confidence >= 0.90: return "Confiança do modelo é **alta** para esta previsão."
        if confidence >= 0.75: return "Confiança do modelo é **moderada**. Baseada em tendências consistentes."
        return "Confiança do modelo é **baixa**. Use como orientação preventiva devido à volatilidade."
