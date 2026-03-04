# 🚀 Instalação e Ativação do Admin Dashboard

## ✅ Status Atual

O código foi **integrado com sucesso** em `app.py`. Agora é necessário instalar a dependência faltante.

---

## 1️⃣ Instalar psutil

O Health Monitor precisa de `psutil` para coletar métricas do sistema.

### Windows (PowerShell ou CMD)
```bash
# Se você tem ambiente virtual ativado:
pip install psutil

# Ou especificar versão:
pip install psutil==5.9.4
```

### Linux/Mac
```bash
pip install psutil
# ou
pip3 install psutil
```

### Verificar Instalação
```bash
python -c "import psutil; print('✅ psutil instalado com sucesso')"
```

---

## 2️⃣ Atualizar requirements.txt

Adicione `psutil` ao arquivo `requirements.txt`:

```bash
# Abrir requirements.txt e adicionar:
psutil==5.9.4
```

Ou use este comando:
```bash
echo "psutil==5.9.4" >> requirements.txt
```

---

## 3️⃣ Reiniciar o App

Após instalar psutil, reinicie o Flask:

```bash
python app.py
```

**Saída esperada ao iniciar:**
```
...
✅ Metadados Regionais Unificados: 259 localidades.
✅ Motor de Inteligência ST-GAT Ativo.
✅ Health Monitor Inicializado.
✅ Admin Dashboard Registrado em /admin/health
...
 * Running on http://127.0.0.1:5050
```

---

## 4️⃣ Acessar o Dashboard

### URL Principal
```
http://localhost:5050/admin/health
```

### API Endpoints
```
GET  /api/admin/health/summary          # Status completo
GET  /api/admin/health/api-stats        # Stats de API
GET  /api/admin/health/alerts           # Alertas ativas
GET  /api/admin/health/confidence-history  # Histórico de confiança
```

### Teste Rápido
```bash
# Com o app rodando:
curl http://localhost:5050/api/admin/health/summary | jq '.'

# Esperado: JSON com system, api, alerts
```

---

## 🆘 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'psutil'"

**Solução:**
```bash
pip install psutil
pip install --upgrade pip
pip install psutil==5.9.4
```

### ❌ "Health Monitor não disponível"

**Causa:** psutil não instalado ou erro na importação

**Solução:**
1. Instale psutil: `pip install psutil`
2. Reinicie o app: `python app.py`
3. Verifique logs: procure por `✅ Health Monitor Inicializado`

### ❌ "404 Not Found" ao acessar `/api/admin/health/*`

**Causa:** Blueprint não foi registrado

**Solução:**
1. Verifique se `✅ Admin Dashboard Registrado em /admin/health` aparece nos logs
2. Reinicie o app com `python app.py`
3. Aguarde 5 segundos até inicializar completamente

### ❌ "Erro ao atualizar dashboard: SyntaxError: Unexpected token '<'"

**Causa:** API retornando HTML (erro 404) em vez de JSON

**Solução:**
1. Verifique se os endpoints estão respondendo:
```bash
curl -v http://localhost:5050/api/admin/health/summary
```

2. Deve retornar `200 OK` com JSON, não `404 Not Found`

3. Se retornar 404, verifique se o app foi reiniciado após instalar psutil

---

## ✅ Checklist de Sucesso

- [ ] psutil instalado (`pip install psutil`)
- [ ] App reiniciado (`python app.py`)
- [ ] Logs mostram `✅ Health Monitor Inicializado`
- [ ] Logs mostram `✅ Admin Dashboard Registrado`
- [ ] `/api/admin/health/summary` retorna JSON (não erro 404)
- [ ] Dashboard carrega em `http://localhost:5050/admin/health`
- [ ] Métricas aparecem em tempo real (CPU, memória, disco)
- [ ] Alertas funcionam

---

## 📊 Dados Persistidos

O Health Monitor salva dados automaticamente:

```
data/health_metrics.json       # Histórico de métricas
data/health_alerts.json        # Alertas persistidos
data/confidence_history.json   # Histórico de confiança
```

Esses arquivos são criados automaticamente ao rodar o app.

---

## 🔄 Auto-Refresh

O dashboard atualiza automaticamente a cada 30 segundos.

Se quiser mudar este intervalo, edite em `admin_health_dashboard.html` (linha ~501):

```javascript
// Mudar 30000 (30 segundos) para outro valor:
const autoRefreshInterval = 30000; // em milissegundos
```

---

## 📚 Documentação Relacionada

- `DOCUMENTACAO_ADMIN_DASHBOARD.md` - Guia completo do dashboard
- `INTEGRACAO_ADMIN_DASHBOARD.md` - Instruções de integração técnica
- `DOCUMENTACAO_API_REST.md` - Referência de endpoints

---

## 🎯 Próximas Etapas

1. ✅ Instalar psutil
2. ✅ Reiniciar app
3. ✅ Acessar /admin/health
4. ✅ Configurar alertas por email (futuro)
5. ✅ Deploy em produção

---

**Versão:** 2.0  
**Data:** 01 de Março de 2026  
**Status:** ✅ Pronto para Usar

*Se tiver problemas, consulte o troubleshooting acima ou abra uma issue.*
