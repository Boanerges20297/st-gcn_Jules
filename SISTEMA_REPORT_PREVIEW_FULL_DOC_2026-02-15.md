# Documentação Técnica: Orquestrador Estadual Report Preview (ST-GAT)
**Data de Emissão:** 15 de Fevereiro de 2026
**Versão:** 7.5.0 - Unificação Estadual

## 1. Visão Geral da Arquitetura
O sistema Report Preview evoluiu de um modelo monolítico para um **Orquestrador Regional Polimórfico**.
 Ele gerencia três especialistas independentes baseados na arquitetura **DeepSTGAT**, cada um calibrado para a dinâmica criminal específica de sua zona de atuação.

### Componentes Principais:
- **Orchestrator (`Phase4/orchestrator.py`):** O cérebro central que roteia requisições para o especialista correto.
- **Architectures (`Phase4/architectures.py`):** Definições das redes neurais (DeepSTGAT_64 e DeepSTGAT_32).
- **Dashboard (`app.py`):** Interface Flask que consome o Orquestrador e gerencia metadados globais.

---

## 2. Fluxo de Carregamento e Inicialização
Ao subir o servidor (`python app.py`), o sistema executa os seguintes passos:

1.  **Carga de Metadados Globais:** Lê o arquivo `processed_graph_data_global.pkl` para obter a lista unificada de 299 localidades (Bairros de Fortaleza + Cidades RMF + Cidades Interior).
2.  **Inicialização dos Especialistas:** O Orquestrador varre as pastas de modelos e carrega:
    - **Fortaleza (Phase 5):** Arquitetura 64 canais, janela de 30 dias.
    - **RMF (Phase 6):** Arquitetura 32 canais, janela de 30 dias.
    - **Interior (Phase 7):** Arquitetura 32 canais, janela de 45 dias.
3.  **Blindagem de Grafos:** O Dashboard detecta que os grafos globais são None e delega a gestão de adjacência (Geográfica e Conflito) para cada especialista.

---

## 3. Funcionamento Preditivo (30d/45d -> 7d)
O sistema opera em um regime purista de momentum recente:
- **Janela de Observação:** Analisa os últimos 30 dias (Capital/RMF) ou 45 dias (Interior).
- **Horizonte de Predição:** Estima o risco de letalidade (CVLI) para os próximos 7 dias.
- **DNA de Facção (Canal 27):** O canal de temperatura estadual das facções conecta cidades distantes através de lealdades criminais, permitindo que o risco flua entre regiões.

---

## 4. Métricas e Realismo Operacional
Para garantir que o Dashboard seja uma ferramenta útil ao gestor e não apenas um gerador de números altos, implementamos:

- **Dampening (Amortecimento):** Scores brutos acima de 50% são suavizados (`50 + (raw - 50) * 0.85`). Isso evita que o gestor veja 100% de risco levianamente, reforçando a natureza estatística da previsão.
- **Normalização Regional:** Cada especialista normaliza seus scores internamente, garantindo que o "mais violento" de cada região seja destacado proporcionalmente ao seu contexto.
- **Tensão Estadual (Escala 0-10):** A métrica de volatilidade foi convertida para um índice de comando intuitivo. 
    - **0-5:** Estável
    - **5-7.5:** Alerta
    - **7.5-10:** Crítico

---

## 5. Protocolo para Novos Modelos e Expansão
Caso deseje treinar um novo especialista ou atualizar um existente:

1.  **Treinamento:** Use os scripts de treino em `PhaseX/src/`. Garanta que o checkpoint final seja salvo com o nome padrão (ex: `model_fortaleza_final.pth`).
2.  **Migração:** Mova o `.pth` para a respectiva pasta em `models/` e o `.pkl` de dados para `data/processed/`.
3.  **Unificação de Metadados:** Sempre que adicionar uma nova localidade, rode o script `scripts/rebuild_global_metadata.py` para atualizar o arquivo global do Dashboard.
4.  **Reconexão:** O Orquestrador detectará o novo arquivo automaticamente no próximo boot.

---

## 6. Sincronização Frontend-Backend
O match entre o modelo e o mapa é feito via **Normalização Soberana de Nomes**:
- Todos os acentos são removidos.
- Sufixos de AIS (ex: " - AIS 12") são deletados via Regex.
- Isso garante que o risco calculado para "CAUCAIA" sempre caia no polígono "CAUCAIA - AIS 12" do Leaflet.

---
**Autor:** Gemini CLI / Report Preview System
**Status do Sistema:** Operacional Unificado (Fortaleza, RMF, Interior).
