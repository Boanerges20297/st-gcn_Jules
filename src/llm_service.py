import os
import json
import logging
import google.generativeai as genai
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_exogenous_text(text: str) -> List[Dict[str, Any]]:
    """
    Process raw exogenous text using Gemini LLM to extract structured events.

    Args:
        text (str): Raw text (e.g., CIOPS report).

    Returns:
        List[Dict]: A list of event dictionaries with keys:
                    'natureza', 'localizacao_completa', 'bairro', 'municipio', 'resumo'.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        logger.warning("GOOGLE_API_KEY not found. Using mock response.")
        return _mock_response(text)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        Analise o seguinte texto de relatório de segurança pública (formato CIOPS).
        O formato padrão dos dados é: "01 - CODIGO - TIPO DA OCORRENCIA - DESCRIÇÃO - LOCAL - DATA".
        Exemplo: "01 - M20260051891 - HOMICIDIO A BALA - VITIMA LESIONADA... - RUA A, BAIRRO X - 01/01/2025"

        Identifique eventos distintos. Para cada evento, extraia e normalize os seguintes dados em formato JSON:

        - "natureza": O TIPO DA OCORRÊNCIA (geralmente o terceiro campo separado por hifens). Extraia exatamente como está.
        - "localizacao_completa": O endereço mais completo possível encontrado (geralmente após a descrição).
        - "bairro": O nome do bairro (inferido ou explícito no endereço).
        - "municipio": O município (ex: FORTALEZA, CAUCAIA). Se não explícito, tente inferir pelo bairro.
        - "resumo": Uma descrição curta baseada no campo DESCRIÇÃO.
        - "raw_text": O trecho original do texto que gerou este evento.

        Texto:
        {text}

        Responda APENAS com uma lista JSON válida de objetos.
        """

        response = model.generate_content(prompt)

        # Simple cleanup to ensure JSON parsing
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        events = json.loads(content)

        # Normalize keys just in case
        normalized_events = []
        for evt in events:
            normalized_events.append({
                'natureza': evt.get('natureza', 'DESCONHECIDO'),
                'localizacao_completa': evt.get('localizacao_completa', ''),
                'bairro': evt.get('bairro', ''),
                'municipio': evt.get('municipio', 'FORTALEZA'), # Default to Fortaleza if missing
                'resumo': evt.get('resumo', ''),
                'raw': evt.get('raw_text', '')
            })

        return normalized_events

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        # Fallback to mock/empty on error? Or re-raise?
        # For this context, returning empty list or mock is safer than crashing app.
        return _mock_response(text)

def _mock_response(text: str) -> List[Dict[str, Any]]:
    """
    Mock response for testing or when API key is missing.
    It tries to do a very basic parse or returns a dummy event if text is present.
    """
    events = []
    lines = text.strip().split('\n')
    for line in lines:
        # Regex heuristic for "01 - CODE - TYPE - DESC - LOC"
        # Split by " - "
        parts = line.split(" - ")
        if len(parts) >= 3:
            # Try to grab the 3rd element as Nature if available, else 2nd
            # Format: Index - Code - Nature - Desc - Loc
            # Example: 01 - M123 - HOMICIDIO - ...
            nature = "EVENTO"
            loc = "LOCAL DESCONHECIDO"

            if len(parts) >= 3:
                nature = parts[2].strip()

            if len(parts) >= 5:
                loc = parts[4].strip()
            elif len(parts) >= 4:
                loc = parts[3].strip()

            events.append({
                'natureza': nature,
                'localizacao_completa': loc,
                'bairro': '',
                'municipio': 'FORTALEZA',
                'resumo': f"{nature} - {loc}",
                'raw': line
            })
    return events
