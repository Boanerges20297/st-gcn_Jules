import requests
import json
import logging
import re
import os

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """
    Cliente de LLM Local otimizado para interagir com o Ollama ou endpoints compatíveis.
    Especializado em retornar e parsear JSONs válidos de forma cirúrgica e com velocidade.
    """
    def __init__(self, model_name: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434", timeout: int = 300):
        self.model_name = os.environ.get("LOCAL_LLM_MODEL", model_name)
        self.base_url = os.environ.get("LOCAL_LLM_URL", base_url)
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.1, response_format: str = None) -> str:
        """
        Executa uma chamada de geração direta para o Ollama local.
        Levanta erros reais e impede fallbacks artificiais.
        """
        url = f"{self.base_url.rstrip('/')}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 256
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if response_format == "json":
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result_json = response.json()
            content = result_json.get("response", "").strip()
            if not content:
                raise RuntimeError("Ollama retornou resposta vazia")
            return content
        except requests.exceptions.RequestException as e:
            logger.error("Falha ao consultar Ollama local: %s", e)
            raise RuntimeError(f"Falha ao consultar Ollama local: {e}") from e

    def parse_json_safely(self, text: str) -> dict:
        """
        Extrai e formata um JSON válido de forma resiliente usando expressões regulares.
        Útil para capturar respostas de LLMs que adicionam marcações Markdown.
        """
        if not text:
            return {}
            
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
            
        try:
            return json.loads(s)
        except Exception:
            pass
            
        # Regex para capturar tudo que está entre chaves
        m = re.search(r"(\{.*\})", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
                
        logger.warning(f"Não foi possível parsear a resposta do agente local como JSON. Texto bruto: {text[:200]}")
        return {}
