# 🚀 Guia de Getting Started - REPORT PREVIEW

## ⏱️ Inicie em 5 Minutos

### Pré-requisitos
- **Python 3.8+** instalado
- **Git** instalado
- **Chave de API do Google Gemini** (obtida em [Google AI Studio](https://aistudio.google.com))
- **Espaço em disco:** ~2GB

---

## 1. Instalação Rápida

### 1.1 Clonar o Repositório

```bash
git clone https://github.com/seu-org/st-gcn-jules.git
cd st-gcn_jules
```

### 1.2 Criar Ambiente Virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Instalar Dependências

```bash
pip install -r requirements.txt
```

*⏱️ Tempo esperado: 2-3 minutos (depende de conexão)*

### 1.4 Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Chave de API do Google Gemini (obrigatório)
GOOGLE_API_KEY=sua_chave_aqui

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Servidor
HOST=127.0.0.1
PORT=5050

# Logging
LOG_LEVEL=INFO
```

**Como obter `GOOGLE_API_KEY`:**
1. Acesse [https://aistudio.google.com](https://aistudio.google.com)
2. Clique em "Get API Key"
3. Copie a chave gerada
4. Cole no `.env` acima

---

## 2. Primeiro Teste (2 min)

### 2.1 Iniciar o Servidor

```bash
python app.py
```

**Saída esperada:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5050
 * Press CTRL+C to quit
```

### 2.2 Abrir Dashboard

Acesse: **http://localhost:5050**

Você verá:
- 🗺️ Mapa com bairros coloridos (Fortaleza, RMF, Interior)
- 📊 Painel de métricas (Temperatura do Estado, Confiança, Top 10)
- 💾 Área para ingestão de eventos

---

## 3. Estrutura do Projeto

```
st-gcn_jules/
├── app.py                           # Servidor Flask (main entry point)
├── requirements.txt                 # Dependências Python
├── .env                            # Variáveis de ambiente
├── Dockerfile                       # Containerização
├── docker-compose.yml              # Orquestração com Docker
│
├── data/                           # Armazenamento de dados
│   ├── processed/                  # Dados históricos GeoJSON
│   ├── exogenous_events.json       # Eventos em tempo real (cache)
│   └── archives/                   # Eventos com >7 dias
│
├── models/                         # Pesos treinados
│   ├── fortaleza_model.pth
│   ├── rmf_model.pth
│   └── interior_model.pth
│
├── src/
│   ├── core/
│   │   ├── orchestrator.py         # Gerenciador de modelos regionais
│   │   ├── architectures.py        # Redes Neurais (ST-GAT)
│   │   ├── data_processing.py      # ETL e construção de tensores
│   │   └── efficiency_monitor.py   # Monitor de backtesting
│   │
│   ├── llm_service.py              # Integração Google Gemini
│   └── explanation_generator.py    # Motor de explicabilidade
│
├── templates/                      # Frontend HTML/JavaScript
│   ├── index.html                  # Dashboard principal
│   └── connections.html            # Mapa de facções
│
├── logs/                          # Logs de treino e operação
│   ├── training_*.log
│   └── rankings/                  # Relatórios diários
│
└── docs/
    └── DOCUMENTACAO_*.md          # Documentação complementar
```

---

## 4. Primeiros Passos na API

### 4.1 Testar Endpoint de Risco

```bash
curl http://localhost:5050/api/risk | jq '.'
```

**O que você verá:**
- Scores de risco para cada bairro (0-100%)
- Nível de confiança do modelo (0-100%)
- Temperatura do Estado (métrica agregada)

### 4.2 Processar um Evento Policial

```bash
curl -X POST http://localhost:5050/api/exogenous/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "AÇÃO POLICIAL em Bom Jardim: Prisão qualificada, apreensão de 2 fuzis. 14:30h",
    "source": "CIOPS"
  }'
```

**O que acontece:**
1. Texto enviado ao Gemini LLM
2. Extração de: data, hora, bairro, natureza, itens apreendidos
3. Impacto calculado e aplicado ao mapa de risco
4. Resposta retorna mudanças de risco

### 4.3 Simular um Cenário

```bash
curl -X POST http://localhost:5050/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "suppression",
    "location_id": 1,
    "teams_deployed": 5,
    "hours_duration": 12
  }'
```

---

## 5. Variáveis de Ambiente Detalhadas

### 5.1 Obrigatórias

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `GOOGLE_API_KEY` | Chave da API Gemini | `AIzaSy...` |

### 5.2 Recomendadas

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `FLASK_ENV` | Ambiente (development/production) | `development` |
| `FLASK_DEBUG` | Modo debug ativo? | `True` |
| `HOST` | IP de escuta | `127.0.0.1` |
| `PORT` | Porta do servidor | `5050` |
| `LOG_LEVEL` | Nível de logs | `INFO` |

### 5.3 Exemplo `.env` Completo

```env
# API Keys
GOOGLE_API_KEY=AIzaSyC...sua_chave_completa...

# Flask
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_APP=app.py

# Servidor
HOST=127.0.0.1
PORT=5050

# Banco de Dados
DATA_DIR=./data
MODELS_DIR=./models

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Performance
CACHE_ENABLED=True
CACHE_EXPIRY_MINUTES=60

# Segurança (futuro)
JWT_SECRET=your_jwt_secret_key
RATE_LIMIT_ENABLED=False
```

---

## 6. Verificação de Saúde do Sistema

### 6.1 Health Check

```bash
curl http://localhost:5050/api/model-update-status | jq '.'
```

**Resposta esperada:**
```json
{
  "status": "idle",
  "model_version": "2.0",
  "last_update": "2026-03-01T10:30:00Z",
  "confidence_global": 0.87
}
```

### 6.2 Verificar Modelos Carregados

Ao iniciar, você verá no console:
```
✅ Carregando modelos regionais...
   ✓ Fortaleza (127 bairros)
   ✓ RMF (43 bairros)
   ✓ Interior (89 bairros)
✅ Iniciando Monitor de Eficiência (próxima avaliação em 7 dias)
```

---

## 7. Docker (Opcional)

Se preferir usar **Docker Compose** para evitar instalar dependências:

### 7.1 Iniciar com Docker

```bash
docker-compose up -d
```

**Logs:**
```bash
docker-compose logs -f app
```

**Parar:**
```bash
docker-compose down
```

### 7.2 Arquivo `docker-compose.yml`

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5050:5050"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
```

---

## 8. Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'src'"

**Solução:**
```bash
# Certifique-se de estar na raiz do projeto
cd st-gcn_jules

# Reinstale dependências
pip install -r requirements.txt

# Execute novamente
python app.py
```

### ❌ "GOOGLE_API_KEY not found"

**Solução:**
1. Verifique se o arquivo `.env` existe
2. Confirme que `GOOGLE_API_KEY` está preenchida
3. Teste a chave em [https://aistudio.google.com](https://aistudio.google.com)

### ❌ "Address already in use (port 5050)"

**Solução:**
```bash
# Opção 1: Mudar a porta no .env
PORT=5051

# Opção 2: Matar processo na porta 5050 (Windows)
netstat -ano | findstr :5050
taskkill /PID <PID> /F

# Opção 3: Linux/Mac
lsof -i :5050
kill -9 <PID>
```

### ❌ "Erro ao carregar modelos (.pth)"

**Solução:**
```bash
# Verifique se os arquivos existem
ls models/

# Se não existirem, baixe de: (repositório ou servidor de backup)
# Ou retraia os modelos a partir dos dados históricos
python scripts/train_models.py
```

---

## 9. Próximos Passos

### ✅ Depois de confirmar que tudo funciona:

1. **Explorar Dashboard**
   - Navegue pela interface
   - Entenda cores e métricas
   - Teste zoom e filtros de região

2. **Ingerir Dados de Teste**
   - Use a interface para colar eventos policiais
   - Veja mudanças refletirem em tempo real

3. **Usar a API**
   - Automatize ingestão de eventos
   - Integre com sistemas existentes (CIOPS, etc)
   - Crie alertas customizados

4. **Ler Documentação Complementar**
   - [DOCUMENTACAO_SISTEMA_MASTER.md](./DOCUMENTACAO_SISTEMA_MASTER.md) - Visão técnica completa
   - [DOCUMENTACAO_API_REST.md](./DOCUMENTACAO_API_REST.md) - Endpoints detalhados
   - [DOCUMENTACAO_CANAIS_REPORT_PREVIEW.md](./DOCUMENTACAO_CANAIS_REPORT_PREVIEW.md) - Variáveis técnicas

---

## 10. Comandos Úteis

```bash
# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar pacote adicional
pip install <package_name>

# Gerar requirements.txt (se adicionar pacotes)
pip freeze > requirements.txt

# Verificar porta ativa
netstat -an | grep 5050  # Linux/Mac
netstat -ano | findstr :5050  # Windows

# Limpar cache de modelos Python
find . -type d -name __pycache__ -exec rm -r {} +

# Listar logs recentes
ls -lt logs/

# Tail logs em tempo real
tail -f logs/app.log
```

---

## 11. Configuração de IDE (VS Code)

### 11.1 Abrir Projeto
```bash
code .
```

### 11.2 Instalar Extensões Recomendadas
- Python (Microsoft)
- Pylance
- Flask Snippets
- REST Client (para testar endpoints)

### 11.3 Configurar Debugger

Crie `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Flask",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "env": {
        "FLASK_APP": "app.py",
        "FLASK_ENV": "development"
      },
      "args": ["run"],
      "jinja": true
    }
  ]
}
```

---

## 12. Checklist Final

- [ ] Python 3.8+ instalado
- [ ] Repositório clonado
- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo `.env` criado com `GOOGLE_API_KEY`
- [ ] Servidor iniciado (`python app.py`)
- [ ] Dashboard acessível em http://localhost:5050
- [ ] `/api/risk` retorna JSON válido
- [ ] Evento de teste processado com sucesso

✅ **Se todos os itens estão marcados, você está pronto para usar REPORT PREVIEW!**

---

**Última atualização:** 01 de Março de 2026  
**Versão:** 2.0
