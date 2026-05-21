# Analise Gemini CLI - Fortaleza

Gerado em: 2026-05-20 20:04:14
Escopo: Fortaleza
Modelo Gemini CLI: gemini-2.5-flash
CSV fonte: outputs/hermes/risk_fortaleza_latest.csv
Historico CSV: outputs/hermes/history/gemini_input_fortaleza_20260520_200336.csv
CSV convergencia: outputs/hermes/dados_status_enriquecido_14d_latest.csv
Historico convergencia: outputs/hermes/history/gemini_convergence_input_fortaleza_20260520_200336.csv

strategic_intent:Gerar uma análise tática e gerencial sobre os bairros mais críticos de Fortaleza, com base no "Hermes Brief de Risco" e nos "dados_status_enriquecido_14d_latest.csv" fornecidos, seguindo a estrutura e as diretrizes estabelecidas.
## 1. Dados até
15/05/2026

## 2. Leitura Rápida
Os bairros mais críticos em Fortaleza, conforme o snapshot, são AEROLANDIA, BARROSO, CURIO, MESSEJANA e BOM JARDIM. O principal fator de risco para o topo da lista de Fortaleza é a atividade recente e a vizinhança, com suporte territorial significativo e registros recentes na janela de 30 dias, especialmente para Aerolândia.

## 3. Padrões Observados
*   **Concentração de Risco Elevado:** Os bairros AEROLANDIA (crítico), BARROSO, CURIO, MESSEJANA, BOM JARDIM e ANCURI (todos altos) dominam o ranking de risco, indicando áreas de alta prioridade.
*   **Drivers de Risco Variados:**
    *   AEROLANDIA: O risco é impulsionado primariamente por "Atividade recente e vizinhança", com suporte territorial de 100% e 2 registros recentes na janela de 30 dias.
    *   BARROSO, CURIO, MESSEJANA, BOM JARDIM, ANCURI, CONJUNTO PALMEIRAS, ALTO ALEGRE, PLANALTO AYRTON SENNA, SIQUEIRA, MONDUBIM, PEDRAS, JOSE WALTER: O driver principal é a "Tensão territorial", muitos com suporte territorial de 100%.
    *   PARQUE IRACEMA e CANINDEZINHO: O "Sinal neural do ST-GAT" é o principal driver, com suporte territorial menor (15% e 50% respectivamente), indicando que o modelo detecta padrões de risco mesmo com menor tensão territorial imediata ou atividade recente tão evidente.
*   **Registros Recentes (30 dias):** AEROLANDIA (2), BARROSO (2), BOM JARDIM (1), ANCURI (1), CONJUNTO PALMEIRAS (1), CANINDEZINHO (1) apresentam registros recentes na janela de 30 dias, o que valida a dinâmica do risco nesses locais. CURIO, MESSEJANA, PARQUE IRACEMA, ALTO ALEGRE, PLANALTO AYRTON SENNA, SIQUEIRA, MONDUBIM, PEDRAS e JOSE WALTER não mostram registros recentes na janela de 30 dias no brief, sugerindo que o risco é mais preditivo com base em tensão territorial ou sinal neural histórico.

## 4. Convergência com dados_status_ENRIQUECIDO (últimos 14 dias)
A análise dos eventos registrados em `dados_status_enriquecido_14d_latest.csv` para Fortaleza, referente aos últimos 14 dias (até 15/05/2026), revela o seguinte:

*   **AEROLANDIA (Rank 1):** Apresenta alta convergência, com aproximadamente 30 eventos de CVP (principalmente roubos a pessoa, coletivo, estabelecimentos, farmácia e um homicídio doloso) registrados nos últimos 14 dias, validando sua posição de risco crítico.
*   **ANCURI (Rank 6):** Mostra convergência, com 8 eventos de CVP (roubos a pessoa e estabelecimentos) nos últimos 14 dias.
*   **BOM JARDIM (Rank 5):** Apresenta convergência, com 5 eventos de CVP (roubos a pessoa, estabelecimentos, supermercado/mercantil, posto de gasolina) nos últimos 14 dias.
*   **BARROSO (Rank 2), CURIO (Rank 3), MESSEJANA (Rank 4), CONJUNTO PALMEIRAS (Rank 7), PARQUE IRACEMA (Rank 8), ALTO ALEGRE (Rank 9), PLANALTO AYRTON SENNA (Rank 10), SIQUEIRA (Rank 11), MONDUBIM (Rank 12), CANINDEZINHO (Rank 13, para Fortaleza), PEDRAS (Rank 14), JOSE WALTER (Rank 15):** Não foram encontrados registros explícitos de eventos nos últimos 14 dias no extrato fornecido do `dados_status_enriquecido_14d_latest.csv` para estes bairros em Fortaleza. Esta ausência de registros diretos nos dados enriquecidos para a maioria dos bairros do ranking superior indica uma evidência inconclusiva de convergência com eventos explícitos *no extrato fornecido*, ou que o risco predito está mais associado a drivers como "Tensão territorial" e "Sinal neural do ST-GAT" que podem não se manifestar em eventos CVP/CVLI imediatos no curto período analisado.

## 5. Pontos de Atenção e Limites
*   **Limitação do Ranking:** A análise foi baseada nos 15 primeiros bairros de Fortaleza devido à limitação do extrato do CSV fornecido.
*   **Confiança dos Dados:** A confiança reportada pelo modelo para os bairros varia, sendo alta para AEROLANDIA (86.0%) e BARROSO (83.7%), mas menor para PARQUE IRACEMA (45.5%). A criticidade para bairros com menor confiança (ex: CURIO 66.4%, MESSEJANA 66.6%) requer validação adicional.
*   **Natureza dos Eventos Enriquecidos:** O CSV de eventos enriquecidos (`dados_status_enriquecido_14d_latest.csv`) concentra-se em "ROUBO A..." (CVP), com poucos registros de "HOMICIDIO DOLOSO" (CVLI). A ausência de CVLI para os bairros do ranking preditivo na maioria dos casos pode ser um ponto de atenção, merecendo investigação sobre a cobertura e tipo de eventos no dado enriquecido.
*   **Divergência/Inconclusividade:** A falta de ocorrências explícitas para grande parte do Top 15 de Fortaleza no dado enriquecido sugere que o risco preditivo pode estar sendo influenciado por fatores que se manifestam de forma diferente dos eventos explicitamente listados nos últimos 14 dias, ou há uma lacuna na granularidade/completude dos dados enriquecidos.

## 6. Recomendações Operacionais
*   **Foco Imediato em Aerolândia:** Dada a alta convergência do ranking preditivo com a atividade real de CVP e CVLI no bairro, Aerolândia deve ser prioridade máxima para alocação de recursos e patrulhamento intensivo.
*   **Validação Territorial para Bairros Top:** Para BARROSO, CURIO, MESSEJANA e BOM JARDIM, onde a "Tensão territorial" é o driver principal e há menor evidência de eventos recentes no dado enriquecido, é crucial realizar verificações proativas de inteligência local, como levantamento de atritos entre grupos, análise de pontos de vulnerabilidade e relatos da comunidade.
*   **Investigação de "Sinal Neural do ST-GAT":** Para bairros como PARQUE IRACEMA e CANINDEZINHO, onde o sinal neural é proeminente, investigar os padrões temporais e espaciais que o modelo está capturando, buscando anomalias sutis que podem preceder eventos maiores.
*   **Aprimoramento da Coleta/Enriquecimento de Dados:** Avaliar a completude e granularidade do `dados_status_enriquecido_14d_latest.csv` para garantir que todos os eventos relevantes estejam sendo capturados, especialmente CVLI, para uma análise de convergência mais robusta para todos os bairros de alto risco.
