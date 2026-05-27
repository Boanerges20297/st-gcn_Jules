# 🗺️ ROADMAP — Milestone: Exportação de Top 30 Micronodos

## ✅ Fase 1-4: Retreino e Fine-tuning (CONCLUÍDO)
- Ver histórico no log de commits ou STATE.md anterior.

## 🔵 Fase 5: Extração de Inteligência Tática
- [ ] Modificar `tests/Sentinela/sentinela_inference.py` para calcular top 30.
- [ ] Implementar mapeamento de micronodo para bairro/regional.
- [ ] Validar saída do CSV/JSON.

## 🔵 Fase 6: Integração com Pipeline de Exportação
- [ ] Identificar local de salvamento do pacote de screenshots.
- [ ] Automatizar a cópia/inclusão do `top_30_micronodes.csv` no pacote.
- [ ] Teste de ponta a ponta: Inferência -> Exportação -> Verificação.

## 🟣 Fase 6.1: Sistema de Agente de IA Local de Background (INSERTED)
- [x] Conceber arquitetura multi-agente local leve (1 Gerente Geral + 3 Especialistas)
- [ ] Implementar integração com LLM local leve especializada em análise de dados (ex: Llama-3-8B-Lexi / Phi-3 via Ollama/Llama.cpp)
- [ ] Desenvolver Agente 1: Gerente Geral (Interface única de controle e comunicação)
- [ ] Desenvolver Agente 2: Especialista em Calibração de Pesos (Ajustes de hiperparâmetros e pesos cirúrgicos)
- [ ] Desenvolver Agente 3: Especialista em Interação do Usuário (Explicações e comunicação em português fluido)
- [ ] Desenvolver Agente 4: Especialista em Análise de Dados Complexos (Raciocínio analítico sobre tendências criminais)
- [ ] Blindar comunicação: especialistas só respondem ao Gerente Geral (Malha Fechada)
- [ ] Criar testes unitários e integrar ao backend Flask

## 🟣 Fase 7: Operação Telegram + Report Preview em VPS Hostinger
- [ ] Definir arquitetura alvo para Ubuntu com API Flask, Gemini CLI e MemPalace.
- [ ] Planejar sincronização segura de artefatos (`data/`, `models/`, `outputs/`) do ambiente local para a VPS.
- [ ] Substituir dependências de PowerShell/Hermes por wrappers Linux e diretivas MemPalace.
- [ ] Planejar hardening operacional: systemd, segredos, observabilidade, rollback e validação ponta a ponta.

---
*Próximo passo sugerido para infraestrutura operacional: /gsd-plan-phase 7*
