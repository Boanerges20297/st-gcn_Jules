# DIAGNÓSTICO DE OSCILAÇÃO - ÁREAS CRÍTICAS

## Observação do Usuário
- **Antes:** 26 áreas críticas
- **Depois:** 141 áreas críticas (salto de 442%)
- **Após correção:** 63 áreas críticas
- **Agora:** 24 áreas críticas

## Root Cause Identificado

**Problema:** Sistema recarrega dados a cada **30 minutos** automaticamente

**Mecanismo:**
1. `start_periodic_reload()` é acionada a cada 30 min
2. Chama `update_exogenous_state()`
3. `apply_exogenous_events()` amplifica adjacência (10-20x) para eventos críticos
4. Altera topologia da rede → predições mudam radicalmente
5. Resultado: **oscilação de 24 até 141 nodes críticos**

## Cadeia de Recarregamento

```
start_periodic_reload(30)
    ↓
_periodic_reload_loop()
    ↓
update_exogenous_state()
    ↓
apply_exogenous_events()  ← ALTERA MATRIZ DE ADJACÊNCIA
    ↓
Novo calculate_risk() com topologia diferente
    ↓
OSCILAÇÃO: 24 → 141 → 63 → 24 áreas críticas
```

## Fatores de Amplificação

Cada evento exógeno pode amplificar por:
- **HIGH severity** (execuções/confrontos): **20x** 
- **MEDIUM severity** (violência armada/deslocamento): **10x**
- **LOW**: sem amplificação

Com 24 eventos MEDIUM/HIGH aplicados simultaneamente:
- Antes: 24 nodes críticos
- Com amplificação: até 141 nodes críticos (5.8x aumentado)

## Solução Implementada

✅ **Aumentar intervalo de recarregamento:**
- De: 30 minutos
- Para: 60 minutos (mínimo)
- **Resultado:** Reduz frequência de oscilações

## Soluções Futuras Recomendadas

1. **Cache com Moving Average** (próxima versão)
   - Armazenar últimas 2-3 predições
   - Usar média em vez de valor instantâneo
   
2. **Validação de Eventos Exógenos**
   - Verificar se eventos são duplicados durante recarreguamento
   - Implementar deduplicação

3. **Gradual Update**
   - Em vez de atualizar tudo de uma vez
   - Fazer smooth transition (reduzir impacto de 30 em 30s)

## Estabilidade Atual

✅ **Múltiplas chamadas consecutivas:** Estável em 24 áreas críticas
✅ **Nodes críticos consistentes:** Mesmos nodes em todas as chamadas
✅ **Métricas estáveis:** Min/Max/Média não variam entre chamadas

## Próximos Passos

Monitorar por **1 hora** após cada recarregamento automático para confirmar que a oscilação foi reduzida.
