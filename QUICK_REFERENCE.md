# 🚀 Quick Reference - REPORT PREVIEW

## 📍 Documentação

### Arquivos Principais (Ler Primeiro)
```
1. DOCUMENTACAO_SISTEMA_MASTER.md      → Visão técnica completa
2. GETTING_STARTED.md                  → Setup em 5 minutos
3. DOCUMENTACAO_API_REST.md            → Endpoints com exemplos
4. DOCUMENTACAO_ADMIN_DASHBOARD.md     → Dashboard e alertas
```

### Referência Rápida
```
DOCUMENTACAO_INDEX.md                  → Índice e roteiros
GLOSSARIO_TECNICO.md                   → Termos A-Z
DOCUMENTACAO_PERFORMANCE_ESCALABILIDADE.md → Limites e scaling
INTEGRACAO_ADMIN_DASHBOARD.md          → Como integrar código novo
```

---

## 🎯 Quick Start

### 1. Instalar (5 min)
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate (Windows)
pip install -r requirements.txt
pip install psutil  # Para health monitor
```

### 2. Configurar (.env)
```env
GOOGLE_API_KEY=sua_chave_gemini_aqui
FLASK_ENV=development
HOST=127.0.0.1
PORT=5050
```

### 3. Rodar
```bash
python app.py
```

### 4. Acessar
- **Dashboard:** http://localhost:5050
- **API de Risco:** http://localhost:5050/api/risk
- **Admin Health (futuro):** http://localhost:5050/admin/health

---

## 📡 Endpoints Principais

### Risco
```bash
GET /api/risk
# → Scores de risco para todos bairros
```

### Eventos
```bash
POST /api/exogenous/parse
# Body: {"text": "Homicídio em Bom Jardim, 22:15h"}
# → Processa com Gemini, atualiza risco
```

### Simulação
```bash
POST /api/simulate
# Body: {"action_type": "suppression", "location_id": 1, "teams_deployed": 5}
# → Projeta impacto de ações
```

### Explicação
```bash
GET /api/explain/1
# → Justifica risco do bairro 1
```

---

## 🛡️ Admin Dashboard (Novo)

### Acessar
```bash
# Após integrar código:
http://localhost:5050/admin/health
```

### Métricas Monitoradas
- ✅ CPU, Memória, Disco
- ✅ Latência P95 da API
- ✅ Taxa de erro
- ✅ Confiança do modelo (P10, P20)
- ✅ Alertas automáticos

### Endpoints Admin
```bash
GET  /api/admin/health/summary
GET  /api/admin/health/api-stats
GET  /api/admin/health/alerts
GET  /api/admin/health/confidence-history
POST /api/admin/health/action
```

---

## 📊 Cores do Mapa de Risco

```
🟦 Azul    <50%   → BAIXO        (patrulhas normais)
🟧 Laranja 50-79% → MODERADO     (monitoramento)
🟥 Vermelho 80-89% → ALTO        (prioridade)
⬛ Vinho   ≥90%   → CRÍTICO      (intervenção)
```

---

## 🔑 Siglas Importantes

| Sigla | Significado |
|-------|------------|
| P10 | Precision top 10 bairros |
| P20 | Precision top 20 bairros |
| ST-GAT | Spatial-Temporal Graph Attention Network |
| CVLI | Crimes Violentos Letais Intencionais |
| CIOPS | Centro Informações Operacionais Polícia |
| RMF | Região Metropolitana Fortaleza |
| LLM | Large Language Model (Gemini) |

---

## 🧪 Testes Rápidos

### 1. Health Check
```bash
curl http://localhost:5050/api/model-update-status | jq '.'
```

### 2. Obter Risco
```bash
curl http://localhost:5050/api/risk | jq '.regions.fortaleza.neighborhoods[0]'
```

### 3. Processar Evento
```bash
curl -X POST http://localhost:5050/api/exogenous/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Prisão qualificada em Bom Jardim"}' | jq '.impact'
```

### 4. Admin Health (após integração)
```bash
curl http://localhost:5050/api/admin/health/summary | jq '.system'
```

---

## 🔧 Troubleshooting Rápido

### "ModuleNotFoundError: No module named 'xxx'"
```bash
pip install -r requirements.txt
pip install psutil
```

### "GOOGLE_API_KEY not found"
```bash
# Criar/editar .env:
GOOGLE_API_KEY=AIzaSy...
```

### "Port already in use"
```bash
# Mudar em .env:
PORT=5051
```

### "Permission denied" (data/)
```bash
mkdir -p data/archives
chmod 755 data
```

---

## 📚 Aprender Mais

### Conceitos Técnicos
- [Glossário Técnico](GLOSSARIO_TECNICO.md) → Termos A-Z
- [Sistema Master](DOCUMENTACAO_SISTEMA_MASTER.md) → Arquitetura

### APIs & Integração
- [API REST](DOCUMENTACAO_API_REST.md) → 10+ endpoints
- [Getting Started](GETTING_STARTED.md) → Setup completo

### Admin & Operações
- [Admin Dashboard](DOCUMENTACAO_ADMIN_DASHBOARD.md) → Monitoramento
- [Performance](DOCUMENTACAO_PERFORMANCE_ESCALABILIDADE.md) → Limites

### Integração de Código
- [Integração](INTEGRACAO_ADMIN_DASHBOARD.md) → Como integrar componentes novos

---

## 🎯 Próximas Ações

### Para Iniciantes
1. Ler GETTING_STARTED.md
2. Rodar `python app.py`
3. Acessar http://localhost:5050
4. Testar `/api/risk`

### Para Desenvolvedores
1. Ler DOCUMENTACAO_API_REST.md
2. Estudar DOCUMENTACAO_SISTEMA_MASTER.md
3. Testar endpoints com curl/Postman
4. Integrar componentes novos

### Para Admins/DevOps
1. Ler DOCUMENTACAO_ADMIN_DASHBOARD.md
2. Integrar health_monitor (INTEGRACAO_ADMIN_DASHBOARD.md)
3. Acessar `/admin/health`
4. Configurar alertas

---

## 🆘 Contato & Suporte

- **Documentação:** Eng. IA
- **Admin Dashboard:** DevOps
- **Operações:** Gestor Segurança
- **API/Integração:** Dev Backend

---

## 💾 Arquivo de Dados

| Arquivo | Localização | Descrição |
|---------|-------------|-----------|
| Modelos | `models/*.pth` | Pesos ST-GAT treinados |
| Dados Históricos | `data/processed/` | 120 dias de histórico |
| Eventos Atuais | `data/exogenous_events.json` | Últimos 7 dias |
| Arquivo Morto | `data/archives/` | Eventos > 7 dias |
| Logs | `logs/` | Treino e operação |
| Health | `data/health_metrics.json` | Métricas do sistema |

---

## 📋 Checklist de Setup

- [ ] Python 3.8+ instalado
- [ ] Repositório clonado
- [ ] Ambiente virtual criado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] psutil instalado (`pip install psutil`)
- [ ] .env criado com GOOGLE_API_KEY
- [ ] Pasta data/archives criada
- [ ] Servidor roda (`python app.py`)
- [ ] `/api/risk` retorna JSON válido
- [ ] Dashboard acessível em localhost:5050

---

## ⏱️ Horários Recomendados

| Tarefa | Frequência | Hora |
|--------|-----------|------|
| Backtesting | 1x/semana | Seg 02:00 |
| Arquivamento | 1x/dia | Diário 03:00 |
| Treino (Retreining) | Sob demanda | - |
| Limpeza de cache | Semanal | Dom 04:00 |
| Relatório de saúde | Diário | Seg-Sex 09:00 |

---

## 🎓 Recursos Online

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph Neural Networks
- [Flask Documentation](https://flask.palletsprojects.com/) - Web Framework
- [Google Gemini API](https://ai.google.dev/) - LLM
- [Chart.js](https://www.chartjs.org/) - Gráficos

---

## 📱 Mobile/Responsivo

Dashboard e API são mobile-friendly:
```
✅ Dashboard responde em todos dispositivos
✅ API retorna JSON (qualquer cliente)
✅ Cards adaptáveis em mobile
✅ Gráficos redimensionam
```

---

**Versão:** 2.0  
**Atualizado:** 01 de Março de 2026  
**Status:** ✅ Pronto para Uso

*Guarde este documento para referência rápida!*
