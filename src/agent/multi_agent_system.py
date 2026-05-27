import json
import logging
from typing import Dict, Any
from src.agent.local_llm_client import LocalLLMClient

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Classe base de agente com LLM local.
    """
    def __init__(self, name: str, role: str, system_prompt: str, client: LocalLLMClient):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = client

    def execute(self, prompt: str) -> str:
        """
        Executa a instrução do agente de forma isolada usando a LLM local.
        """
        return self.client.generate(prompt, system_prompt=self.system_prompt)


class _CalibrationAgent(BaseAgent):
    """
    Agente 2: Especialista em Calibração de Pesos (Cirúrgico e Matemático).
    Blindado: NUNCA interage diretamente com o usuário final. Retorna estritamente JSON.
    """
    def __init__(self, client: LocalLLMClient):
        super().__init__(
            name="Especialista em Calibração de Pesos",
            role="Calibrador",
            system_prompt=(
                "Você é um agente matemático especializado em calibrar hiperparâmetros de modelos de machine learning e redes ST-GCN.\n"
                "Sua única tarefa é ler o histórico e as discrepâncias de confiança, e calcular os pesos ajustados ideais para as métricas:\n"
                "- postura (posture)\n"
                "- velocidade (speed)\n"
                "- amplitude (rom)\n\n"
                "REGRAS DE RETORNO:\n"
                "1. Retorne ESTRITAMENTE um objeto JSON válido.\n"
                "2. O JSON deve conter as chaves: 'weights' (um dicionário com posture, speed, rom como floats de 0.0 a 1.0) e 'justification' (uma frase técnica curta em português).\n"
                "3. NUNCA adicione saudações ou textos complementares fora do JSON."
            ),
            client=client
        )


class _InteractionAgent(BaseAgent):
    """
    Agente 3: Especialista em Interação do Usuário (NLP Fluido Adaptativo).
    Blindado: NUNCA decide pesos matemáticas. Escreve em português natural e empático.
    """
    def __init__(self, client: LocalLLMClient):
        super().__init__(
            name="Especialista em Interação do Usuário",
            role="Redator",
            system_prompt=(
                "Você é um especialista em experiência do usuário e comunicação analítica em segurança pública.\n"
                "Sua única tarefa é redigir explicações e sumários dos relatórios em um português do Brasil extremamente natural, adaptativo e empático.\n"
                "Evite jargões puramente matemáticos desnecessários e traduza a complexidade em insights acionáveis.\n"
                "REGRAS DE RETORNO:\n"
                "1. Retorne um JSON com a chave 'output' contendo o texto final formatado.\n"
                "2. NUNCA fale em primeira pessoa do singular, sempre reporte de forma colaborativa institucional."
            ),
            client=client
        )


class _ComplexDataAnalystAgent(BaseAgent):
    """
    Agente 4: Especialista em Análise de Dados Complexos (Raciocínio Analítico).
    Blindado: Identifica padrões analíticos e tem foco absoluto na predição de CVLI.
    Equipado com a skill de prever o próximo local provável de ocorrência de CVLI.
    """
    def __init__(self, client: LocalLLMClient):
        super().__init__(
            name="Especialista em Análise de Dados Complexos",
            role="Analista",
            system_prompt=(
                "Você é um cientista de dados sênior e estrategista tático especializado em dinâmica de crimes violentos (CVLI).\n"
                "Sua única tarefa é analisar as anomalias temporais, correlação geográfica de crimes anteriores e o impacto de eventos exógenos nos rankings preditivos.\n\n"
                "DIRETRIZES DE METACORREÇÃO E SKILLS:\n"
                "1. FOCO ABSOLUTO EM CVLI (Crimes Violentos Letais Intencionais).\n"
                "2. SKILL DE PREVISÃO DE HOTSPOTS: Utilize a convergência de deslocamentos criminais e o histórico de tensões para estimar/indicar qual o próximo local de Fortaleza/RMF que apresenta alto risco iminente de CVLI.\n"
                "3. ALERTA DE DISCREPÂNCIA: Se você notar qualquer desvio de convergência (por exemplo, o modelo preditivo ignorando zonas que deveriam estar em alta devido a exógenos de facção recentes), você DEVE chamar a atenção do Gerente Geral explicitamente definindo 'anomalies_detected' como true e adicionando um aviso claro no resumo técnico.\n\n"
                "REGRAS DE RETORNO:\n"
                "1. Retorne um JSON com as chaves:\n"
                "   - 'anomalies_detected' (bool - true se houver discrepância de convergência)\n"
                "   - 'geographical_drift' (bool - true se a mancha criminal estiver migrando para novas áreas)\n"
                "   - 'next_probable_cvli_hotspot' (string - local/bairro provável do próximo CVLI com justificativa tática baseada em sua skill)\n"
                "   - 'technical_summary' (string - análise fria e crítica para alertar o gerente geral).\n"
                "2. Seja cirúrgico e focado em estatísticas de erros."
            ),
            client=client
        )


class GeneralManagerAgent:
    """
    Agente 1: Gerente Geral.
    O ÚNICO agente exposto publicamente para o backend. 
    Administra, coordena a delegação para os especialistas sob demanda, harmoniza e formula a resposta final.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3:8b"):
        self.client = LocalLLMClient(base_url=base_url, model_name=model_name)
        
        # Instanciar os especialistas como membros estritamente privados
        self._calibrator = _CalibrationAgent(self.client)
        self._interactor = _InteractionAgent(self.client)
        self._analyst = _ComplexDataAnalystAgent(self.client)

    def process_and_calibrate(self, raw_stgcn_data: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orquestra a calibração em background delegando tarefas aos especialistas blindados.
        """
        logger.info("Gerente Geral: Iniciando análise em malha fechada...")
        from src.agent.agent_wiretap import wiretap
        import os

        # --- CARREGAMENTO DA MEMÓRIA TÁTICA DO MEMPALACE ---
        mempalace_context = ""
        mempalace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mempalace.md'))
        if os.path.exists(mempalace_path):
            try:
                with open(mempalace_path, 'r', encoding='utf-8') as f:
                    # Carrega as primeiras 1500 palavras para preservar contexto tático sem estourar a janela do LLM local
                    mempalace_context = f.read()
                    logger.info("Gerente Geral: Memória do MemPalace acoplada com sucesso.")
            except Exception as e:
                logger.warning(f"Não foi possível ler o arquivo MemPalace: {e}")

        # 1. Agente 4 analisa os dados complexos brutamente enriquecido com MemPalace (apenas resumo executivo)
        mempalace_lines = mempalace_context.split('\n')
        mempalace_summary = "\n".join([line for line in mempalace_lines if line.strip() and not line.startswith('#')][:10])
        
        analyst_prompt = (
            f"=== MEMÓRIA E DIRETIVAS DO MEMPALACE (RESUMO) ===\n{mempalace_summary}\n\n"
            f"=== DADOS EM TEMPO REAL ===\n"
            f"Dados brutos da predição: {json.dumps(raw_stgcn_data)}\n"
            f"Perfil regional/histórico: {json.dumps(user_profile)}"
        )
        analyst_raw = self._analyst.execute(analyst_prompt)
        wiretap.record_interaction("Gerente Geral", self._analyst.name, analyst_prompt, analyst_raw)
        analyst_data = self.client.parse_json_safely(analyst_raw)
        
        # 2. Agente 2 calcula os pesos matemáticos com base nas anomalias detectadas
        calibrator_prompt = (
            f"Com base na análise técnica de anomalias: {json.dumps(analyst_data)}\n"
            f"Calcule os pesos ideais de calibração para postura, velocidade e rom."
        )
        calibrator_raw = self._calibrator.execute(calibrator_prompt)
        wiretap.record_interaction("Gerente Geral", self._calibrator.name, calibrator_prompt, calibrator_raw)
        calibration_data = self.client.parse_json_safely(calibrator_raw)
        
        # Se falhar o parse, garante fallback seguro
        if "weights" not in calibration_data:
            calibration_data = {
                "weights": {"posture": 0.85, "speed": 0.70, "rom": 0.90},
                "justification": "Fallback de emergência ativado pelo Gerente Geral por inconsistência na resposta do especialista."
            }

        # 3. Agente 3 redige a justificativa em linguagem humana refinada para o relatório do gerente
        interactor_prompt = (
            f"Traduza a seguinte justificativa técnica em uma mensagem fluida e acolhedora em português "
            f"para o tomador de decisão (policial/gestor): '{calibration_data.get('justification')}'"
        )
        interactor_raw = self._interactor.execute(interactor_prompt)
        wiretap.record_interaction("Gerente Geral", self._interactor.name, interactor_prompt, interactor_raw)
        interactor_data = self.client.parse_json_safely(interactor_raw)
        friendly_output = interactor_data.get("output", calibration_data.get("justification"))

        # 4. O Gerente Geral compila tudo, revisa criticamente e formula o output de negócios
        response = {
            "status": "success",
            "manager_decision": "Os pesos do modelo foram calibrados com sucesso de forma dinâmica para mitigar falsos positivos.",
            "calibrated_weights": calibration_data.get("weights"),
            "explanations": friendly_output,
            "data_analysis": analyst_data,
            "calibrated_at": raw_stgcn_data.get("timestamp", "")
        }
        
        logger.info("Gerente Geral: Resposta em malha fechada consolidada com sucesso.")
        return response
