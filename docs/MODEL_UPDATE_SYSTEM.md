# 🔄 Sistema de Atualização Automática de Modelos

## Visão Geral

Sistema que monitora mudanças nos dados brutos (`data/raw/`) e dispara automaticamente:
1. **Reprocessamento** dos dados (data_processing.py)
2. **Retreinamento** dos modelos (ST-GCN + Ranking Model)
3. **Atualização visual** no frontend com progresso

---

## 🏗️ Componentes

### 1. Monitor de Atualização (`src/model_update_monitor.py`)

**Responsabilidades:**
- ✅ Detecta mudanças em `data/raw/` (via hash MD5)
- ✅ Dispara pipeline de reprocessamento + retreinamento
- ✅ Mantém estado sincronizado (status, progresso, mensagens)
- ✅ Roda em thread separada (não bloqueia Flask)

**Estados:**
```
idle              → Aguardando mudanças
processing        → Executando data_processing.py
training          → Retreinando modelos
updating_models   → Sincronizando modelos
error             → Erro detectado
```

**Progresso:**
- 0-10%: Preparando
- 10-30%: Reprocessando dados
- 30-40%: Iniciando treinamento
- 40-70%: Retreinando ST-GCN
- 70-90%: Retreinando Ranking Model
- 90-95%: Sincronizando
- 95-100%: Concluído

---

### 2. Backend (app.py)

**Rota adicional:**
```python
@app.route('/api/model-update-status')
```

**Response:**
```json
{
    "status": "training",
    "progress": 65,
    "message": "Retreinando Ranking Model...",
    "error": null,
    "last_check": "2026-02-04T10:30:45.123456",
    "last_update": "2026-02-04T10:25:00.000000"
}
```

**Inicialização:**
```python
if __name__ == "__main__":
    start_monitor(check_interval=300)  # Verifica a cada 5 min
    app.run(...)
```

---

### 3. Frontend (templates/index.html)

**Componentes visuais:**
- ✅ Loading overlay com progresso
- ✅ Barra de progresso animada
- ✅ Porcentagem em tempo real
- ✅ Mensagem descritiva

**Polling:**
- Verifica status a cada **3 segundos**
- Atualiza overlay em tempo real
- Recarrega página quando 95% concluído

**Funções:**
```javascript
showLoadingWithProgress(msg, progress)  // Mostra loading com progresso
checkModelUpdateStatus()                 // Verifica status via API
```

---

## 🔧 Como Funciona

### Fluxo Completo

```
1. Monitor detecta mudanças em data/raw/
   ↓
2. Calcula hash MD5 de todos os arquivos
   ↓
3. Compara com hash anterior (.data_checksum)
   ↓
4. Se diferente:
   ├─ data_processing.py (30s - 2min)
   ├─ train.py ST-GCN (1-5 min)
   ├─ train_ranking_window30_final.py (5-30 min)
   └─ Atualiza models/ com novos arquivos
   ↓
5. Frontend mostra progresso em tempo real
   ↓
6. Ao atingir 95%, recarrega página
   ↓
7. App carrega novos modelos automaticamente
```

---

## 📱 UX do Usuário

### Sem Atualização
- Dashboard funciona normalmente
- Sem interferência no uso

### Com Atualização Detectada
```
┌─────────────────────────────────┐
│    Modelos em Atualização       │
│                                  │
│    [████████░░░░░░░░░░] 65%     │
│                                  │
│  Retreinando Ranking Model...   │
└─────────────────────────────────┘
```

- Overlay aparece automaticamente
- Progresso atualizado a cada 3s
- Sem bloqueio de interação (backdrop apenas)
- Recarregamento automático ao atingir 100%

---

## 🛠️ Configuração

### Intervalo de Verificação

**Em `app.py`:**
```python
start_monitor(check_interval=300)  # segundos
```

Opções recomendadas:
- `60`: Muito frequente (alto CPU)
- `300`: Padrão (5 min)
- `600`: Conservador (10 min)

### Timeout dos Scripts

**Em `model_update_monitor.py`:**
```python
# data_processing.py
timeout=600  # 10 minutos

# ST-GCN training
timeout=3600  # 1 hora

# Ranking training
timeout=1800  # 30 minutos
```

---

## 📊 Monitoramento

### Logs
O sistema registra tudo em stdout:
```
[MONITOR] Iniciado - verificando a cada 300s
[MONITOR] Mudanças detectadas em data/raw/
[MONITOR] Reprocessando dados brutos...
[MONITOR] Retreinando ST-GCN...
[MONITOR] Retreinando Ranking Model...
[MONITOR] Atualização concluída com sucesso!
```

### Arquivo de Checksum
`.data_checksum` - Armazena:
```json
{
    "hash": "a1b2c3d4e5f6...",
    "timestamp": "2026-02-04T10:30:45.123456"
}
```

---

## ⚠️ Tratamento de Erros

Se houver erro durante o processo:

1. **Durante reprocessamento:**
   - Mensagem: "Erro ao reprocessar dados"
   - Status muda para `error`
   - Frontend mostra "Erro detectado"

2. **Durante treinamento:**
   - Mensagem: "Erro ao treinar modelos"
   - Status muda para `error`
   - Frontend mostra erro específico

3. **Timeout:**
   - Se script demore > timeout, é abortado
   - Registra erro e retorna ao `idle`

---

## 🚀 Próximas Melhorias

- [ ] Notificações push ao usuário
- [ ] Log persistente de atualizações
- [ ] Dashboard de histórico
- [ ] Agendamento de manutenção (ex: 02:00 AM)
- [ ] Rollback automático em caso de erro
- [ ] Backup automático de modelos anteriores
- [ ] Execução paralela de retreinamentos

---

## 📝 Troubleshooting

### Monitor não inicia
```bash
# Verificar logs
python app.py  # Procurar por "[MONITOR]"
```

### Atualização muito lenta
- Aumentar `timeout`
- Verificar recursos do sistema (CPU/RAM)
- Considerar executar em GPU

### Frontend não mostra progresso
- Verificar console do navegador (F12)
- Verificar se `/api/model-update-status` retorna 200
- Verificar se CORS está configurado

---

**Criado em**: Fevereiro de 2026  
**Status**: ✅ Production Ready
