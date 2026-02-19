# 🛡️ Report Preview: Sistema de Predição de Risco Criminal (ST-GAT + LLM)

> **Decisão Tática Baseada em Inteligência Artificial Híbrida**

O **Report Preview** é uma plataforma avançada de monitoramento e predição de risco de segurança pública. Utilizando uma arquitetura híbrida que combina **Redes Neurais em Grafos Espaço-Temporais (ST-GAT)** com **Modelos de Linguagem (LLMs)**, o sistema oferece previsões precisas sobre manchas criminais, dinâmicas de facções e alertas de curto prazo para o estado do Ceará.

---

## 🚀 Funcionalidades Principais

*   **Predição Espaço-Temporal:** Utiliza redes ST-GAT para entender não apenas *onde* o crime ocorre, mas *quando* e *como* ele se desloca entre bairros vizinhos.
*   **Análise de Vínculos (Facções):** O modelo considera a "geografia do conflito", onde a proximidade não é apenas física, mas também definida por alianças e rivalidades entre grupos criminosos.
*   **Inteligência em Tempo Real (LLM):** Integração com **Google Gemini** para ler logs policiais (CIOPS), extrair entidades e injetar "eventos de choque" no modelo matemático em tempo real.
*   **Orquestração Regional:** Modelos especialistas distintos para a Capital (Fortaleza), Região Metropolitana e Interior, respeitando as dinâmicas locais de cada área.
*   **Explicação Tática:** Não é uma "Caixa Preta". O sistema gera justificativas em linguagem natural para cada alerta de risco, explicando se a causa é tendência histórica, contágio vizinho ou evento recente.

---

## 🏗️ Arquitetura do Sistema

O sistema é construído em Python e segue um fluxo pipeline robusto:

1.  **Ingestão:** Dados históricos + Eventos de Tempo Real (API).
2.  **Processamento:** Construção de Tensores de 29 Canais (Crimes, Sazonalidade, Eventos Exógenos).
3.  **Core AI:**
    *   **Deep ST-GAT:** Processamento da série temporal e grafo espacial.
    *   **TAG-Bias:** Injeção de viés tático para reatividade imediata a eventos críticos.
4.  **Interface:** Dashboard interativo baseada em mapas para consciência situacional.

---

## 📦 Estrutura do Projeto

```bash
st-gcn_jules/
├── app.py                 # API Gateway (Flask) e Entrypoint
├── data/                  # Armazenamento de dados
│   ├── processed/         # Dados históricos enriquecidos e GeoJSONs
│   └── exogenous_events.json # Eventos de tempo real (Cache)
├── docs/                  # Documentação detalhada
├── models/                # Pesos dos modelos treinados (.pth)
├── src/
│   ├── core/
│   │   ├── architectures.py # Definição das Redes Neurais (ST-GAT)
│   │   ├── orchestrator.py  # Gerenciador dos modelos regionais
│   │   └── data_processing.py # Pipeline de ETL e Tensores
│   ├── llm_service.py       # Integração com Google Gemini
│   └── explanation_generator.py # Motor de explicabilidade
├── templates/             # Frontend do Dashboard (HTML/JS)
└── Dockerfile             # Containerização da aplicação
```

---

## 🛠️ Instalação e Configuração

### Pré-requisitos
*   Python 3.9+
*   Chave de API do Google Gemini (para funcionalidades LLM)

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-org/st-gcn-jules.git
cd st-gcn-jules
```

### 2. Configurar Ambiente Virtual
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```
*(Nota: Certifique-se de instalar as versões corretas do PyTorch e PyTorch Geometric compatíveis com seu CUDA, se aplicável).*

### 4. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz:
```env
GOOGLE_API_KEY=sua_chave_aqui
FLASK_ENV=development
```

---

## ▶️ Como Usar

### Iniciar o Servidor
```bash
python app.py
```
O sistema estará acessível em `http://localhost:5000`.

### Endpoints Principais

*   **Dashboard:** `GET /` - Visualização do mapa de risco.
*   **API de Risco:** `GET /api/risk` - Retorna o JSON com os scores de risco por bairro.
*   **Inserir Evento:** `POST /api/exogenous/parse`
    *   Body: `{"text": "Disparo de arma de fogo no bairro Bom Jardim..."}`
    *   *Processa o texto via LLM e atualiza o risco em tempo real.*
*   **Explicação:** `GET /api/explain?node_id=XXX` - Gera o relatório do porquê aquele bairro está em risco.

---

## 📚 Documentação Complementar

Para detalhes profundos sobre cada componente, consulte a pasta `docs/`:

*   [Fluxo Completo do Sistema](docs/DOCUMENTACAO_FLUXO_COMPLETO.md) - **Recomendado para Engenheiros.**
*   [Guia de Canais e Tensores](DOCUMENTACAO_CANAIS_REPORT_PREVIEW.md) - Entenda as variáveis de entrada.
*   [Monitoramento de Anomalias](docs/ANOMALY_MONITORING_GUIDE.md)
*   [Arquitetura de Referência](docs/ARCHITECTURE_REFERENCE.md)

---

## 🛡️ Licença e Segurança

Este software é de uso restrito para fins de análise de segurança pública e inteligência.
Todos os dados geoespaciais e de ocorrências devem ser tratados com o nível de confidencialidade adequado.

**Desenvolvido por:** Equipe de Inteligência Artificial & Tática.
