# Analise Gemini CLI - Fortaleza

Gerado em: 2026-05-20 17:49:56
Escopo: Fortaleza
Modelo Gemini CLI: gemini-2.5-flash
CSV fonte: outputs/hermes/risk_fortaleza_latest.csv
Historico CSV: outputs/hermes/history/gemini_input_fortaleza_20260520_174914.csv

**1. Dados até:** 15/05/2026
**Fonte:** `outputs/hermes`

**2. Leitura Rápida:**
Os bairros mais críticos em Fortaleza neste snapshot são AEROLANDIA, BARROSO, CURIO, MESSEJANA e BOM JARDIM. AEROLANDIA lidera o ranking com risco de 81.8 e nível crítico, sustentado por atividade recente e vizinhança. BARROSO e CURIO seguem com riscos de 66.3 e 65.5, respectivamente, ambos em nível alto e com a tensão territorial como principal driver. Para o gestor, o peso principal do risco para AEROLANDIA advém da atividade recente e vizinhança, com suporte territorial de 100.0% e 2 registros recentes na janela de 30 dias.

**Top 15 Bairros de Fortaleza:**
1.  **AEROLANDIA** — risco 81.8 | crítico | confiança 86.0%
2.  **BARROSO** — risco 66.3 | alto | confiança 83.7%
3.  **CURIO** — risco 65.5 | alto | confiança 66.4%
4.  **MESSEJANA** — risco 63.0 | alto | confiança 66.6%
5.  **BOM JARDIM** — risco 57.9 | alto | confiança 78.1%
6.  **ANCURI** — risco 56.9 | alto | confiança 78.0%
7.  **CONJUNTO PALMEIRAS** — risco 53.1 | alto | confiança 77.4%
8.  **PARQUE IRACEMA** — risco 53.0 | alto | confiança 45.5%
9.  **ALTO ALEGRE** — risco 45.9 | moderado | confiança 67.7%
10. **PLANALTO AYRTON SENNA** — risco 43.6 | moderado | confiança 67.7%
11. **SIQUEIRA** — risco 41.9 | moderado | confiança 67.7%
12. **MONDUBIM** — risco 38.4 | moderado | confiança 55.9%
13. **CANINDEZINHO** — risco 37.3 | moderado | confiança 53.9%
14. **PEDRAS** — risco 36.9 | moderado | confiança 67.7%
15. **JOSE WALTER** — risco 35.4 | moderado | confiança 67.7%

**3. Padrões Observados:**
*   **Concentração de Risco Elevado:** Os 7 primeiros bairros apresentam nível de risco "alto" ou "crítico", indicando uma concentração de prioridade operacional. AEROLANDIA, BARROSO, CURIO, MESSEJANA, BOM JARDIM, ANCURI e CONJUNTO PALMEIRAS são as áreas de maior preocupação.
*   **Drivers de Risco Predominantes:** "Tensão territorial" é o driver primário ou secundário em quase todos os bairros classificados como alto ou moderado, refletindo sua importância sistêmica. "Atividade recente e vizinhança" é um driver chave para AEROLANDIA, o bairro de maior risco. O "Sinal neural do ST-GAT" também é um fator recorrente.
*   **Suporte Territorial Consistente:** Para a maioria dos bairros de alto risco, o suporte territorial é de 100%, reforçando a base de dados para esses diagnósticos, com exceções pontuais (e.g., PARQUE IRACEMA com 15.0% e CANINDEZINHO com 50.0%).

**4. Pontos de Atenção e Limites:**
*   **Variação na Confiança:** Embora a maioria dos bairros de alto risco tenha confiança superior a 75%, PARQUE IRACEMA (risco alto, rank 8) exibe uma confiança de apenas 45.5%, e CURIO e MESSEJANA (risco alto, ranks 3 e 4) têm confiança em torno de 66%, sugerindo que esses dados podem exigir validação adicional.
*   **Registros Recentes:** Vários bairros no top 15, incluindo CURIO, MESSEJANA, PARQUE IRACEMA, ALTO ALEGRE, PLANALTO AYRTON SENNA, SIQUEIRA, MONDUBIM, PEDRAS e JOSE WALTER, registram 0 ocorrências na janela de 30 dias. Isso indica que o risco elevado nessas áreas pode ser impulsionado por fatores latentes (como tensão territorial ou sinal neural do ST-GAT) em vez de atividade explícita recente, necessitando de investigação para compreender o contexto.
*   **Profundidade da Análise:** A lista apresentada abrange os 15 bairros com maior risco, não os 30 solicitados, devido à granularidade dos dados fornecidos neste extrato.

**5. Recomendações Operacionais:**
*   **Prioridade Imediata:** Direcionar recursos e esforços de inteligência para AEROLANDIA, BARROSO, CURIO, MESSEJANA e BOM JARDIM, que apresentam os riscos mais críticos e altos.
*   **Validação Contextual:** Para todos os bairros listados, verificar eventos recentes, pressão territorial e coerência com a inteligência local. É crucial entender se a ausência de registros recentes em alguns bairros reflete uma diminuição real da atividade ou uma falha na detecção, utilizando informações complementares do campo.
*   **Atenção à Confiança Reduzida:** Realizar uma análise aprofundada nos bairros com níveis de confiança mais baixos, como PARQUE IRACEMA (45.5%), para qualificar ou desqualificar a indicação de risco, buscando dados adicionais para reforçar a avaliação.
*   **Monitoramento de Drivers:** Acompanhar de perto a "tensão territorial" e o "sinal neural do ST-GAT" como fatores que consistentemente predizem o aumento do risco, mesmo na ausência de registros de atividade recente.
*   **Preparação para Atualização:** Considerar estas informações um snapshot tático, com a necessidade de revalidação contínua e integração com novas informações antes da próxima atualização do ranking de risco.
