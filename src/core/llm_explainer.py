"""
LLM Explainer — Explicabilidade opcional via Claude/GPT

Fornece explicação legível de por que um ajuste foi feito.
Roda async (não bloqueia o daemon).
Totalmente opcional — funciona sem dependência de LLM.
"""

import os
import json
import logging
import threading
from typing import Dict, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Tentar importar SDK de LLM
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMExplainer:
    """
    Gera explicações legíveis de decisões de calibração usando LLM.
    Totalmente opcional e async (não bloqueia).
    """

    def __init__(self, provider: str = 'anthropic', enabled: bool = True):
        """
        Args:
            provider: 'anthropic' ou 'openai'
            enabled: Ativar geração de explicações via LLM
        """
        self.provider = provider
        self.enabled = enabled and (HAS_ANTHROPIC or HAS_OPENAI)
        self.client = None
        
        if not self.enabled:
            logger.info("[LLM] ⚠️ LLM Explainer desabilitado (dependências não instaladas)")
            return
        
        if provider == 'anthropic' and HAS_ANTHROPIC:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("[LLM] ✅ Anthropic Claude configurado")
            else:
                self.enabled = False
                logger.warning("[LLM] ⚠️ ANTHROPIC_API_KEY não configurada")
        
        elif provider == 'openai' and HAS_OPENAI:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                self.client = openai
                logger.info("[LLM] ✅ OpenAI GPT configurado")
            else:
                self.enabled = False
                logger.warning("[LJM] ⚠️ OPENAI_API_KEY não configurada")

    def explain_async(self, 
                     adjustment_data: Dict,
                     callback: Optional[Callable] = None) -> threading.Thread:
        """
        Gera explicação em thread separada (não bloqueia).
        
        Args:
            adjustment_data: {region, metric, old_value, new_params, semantic_info, ...}
            callback: Função a chamar quando explicação estiver pronta
        
        Returns:
            Thread (já iniciada)
        """
        if not self.enabled:
            return None
        
        def _generate():
            try:
                explanation = self._generate_explanation(adjustment_data)
                if callback:
                    callback(explanation)
                else:
                    logger.info(f"[LLM] 💭 {explanation}")
            except Exception as e:
                logger.error(f"[LLM] ❌ Erro ao gerar explicação: {e}")
        
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        return thread

    def _generate_explanation(self, adjustment_data: Dict) -> str:
        """Gera explicação via LLM."""
        if not self.enabled or not self.client:
            return ""
        
        prompt = self._build_prompt(adjustment_data)
        
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Modelo rápido e barato
                    max_tokens=300,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text
            
            elif self.provider == 'openai':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    max_tokens=300,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a brief explainer of machine learning model calibration decisions. Keep responses under 50 words."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"[LLM] ❌ Erro na API: {e}")
            return ""

    def _build_prompt(self, adjustment_data: Dict) -> str:
        """Constrói prompt estruturado para o LLM."""
        region = adjustment_data.get('region', '?')
        metric = adjustment_data.get('metric', '?')
        old_value = adjustment_data.get('old_value', 0)
        current_value = adjustment_data.get('current_value', 0)
        threshold = adjustment_data.get('threshold', 0)
        semantic_info = adjustment_data.get('semantic_info', {})
        new_params = adjustment_data.get('new_params', {})
        
        prompt = f"""
Explain briefly (one sentence) why this model calibration adjustment was made:

Region: {region}
Metric: {metric}
Old Confidence: {old_value*100:.1f}%
Current Confidence: {current_value*100:.1f}%
Threshold: {threshold*100:.1f}%
Degradation: {((threshold - current_value) / threshold * 100):.1f}%

Adjustment Made:
- tag_bias_direct: {new_params.get('tag_bias_direct', '?')}
- tension_factor: {new_params.get('tension_factor', '?')}
- norm_neural_weight: {new_params.get('norm_neural_weight', '?')}

Pattern Recognition: {semantic_info.get('source', 'heuristic')}
Similarity: {semantic_info.get('similarity_score', 0):.2f}

Keep response under 20 words. Focus on what the adjustment does (e.g., 'Boosting neural weight to reduce noise in low-confidence regions').
"""
        return prompt.strip()

    def explain_sync(self, adjustment_data: Dict, timeout: float = 2.0) -> str:
        """
        Gera explicação sincronamente com timeout.
        Útil para logging mas com limite de tempo.
        
        Args:
            adjustment_data: Dados do ajuste
            timeout: Timeout em segundos
        
        Returns:
            Explicação (ou string vazia se timeout/erro)
        """
        if not self.enabled:
            return ""
        
        explanation_holder = {'text': ''}
        
        def callback(explanation: str):
            explanation_holder['text'] = explanation
        
        thread = self.explain_async(adjustment_data, callback)
        if thread:
            thread.join(timeout=timeout)
        
        return explanation_holder['text']

    def get_status(self) -> Dict:
        """Retorna status do explainer."""
        return {
            'enabled': self.enabled,
            'provider': self.provider,
            'has_api_key': self.client is not None,
            'message': 'Ready to generate explanations' if self.enabled else 'LLM Explainer disabled'
        }
