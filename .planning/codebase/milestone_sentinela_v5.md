# 🏆 Milestone: Sentinela V5 (Reinforcement Learning Core)

Este documento define os passos críticos para a transição do paradigma **Sentinela V4 (Neural-Manual)** para o **Sentinela V5 (Neural-Autônomo)**, focado em Aprendizado por Reforço.

## 🏁 Prerequisito: Conclusão da V4 (Baseline)
O treino atual da V4 deve atingir os seguintes checkpoints antes de dispararmos a V5:
- [ ] Mínimo de 10 épocas concluídas para Fortaleza.
- [ ] P@10 de validação estabilizada > 45%.
- [ ] P@20 de validação estabilizada > 65%.
- [ ] Registro das "Surpresas" do MemPalace no `training_vault` finalizado.

---

## 🛠️ Próximos Passos (Workflow V5)

### Passo 1: Extração de Embeddings (O "Estado")
Após o treino da V4, utilizaremos o modelo congelado para extrair as representações latentes de cada bairro.
- **Ação:** Criar script `extract_gat_embeddings.py`.
- **Objetivo:** Transformar os 39 canais em um vetor de "Estado Tático" para o Agente RL.

### Passo 2: Definição da Função de Recompensa (Reward Engineering)
A inteligência da V5 depende de uma recompensa que valorize a precisão tática operacional.
- **Fórmula Proposta:** `R = (w1 * Hit@10) + (w2 * Δ_Rank) - (w3 * False_Alarm)`.
- **Foco:** Recompensar o agente quando ele "liga" a memória no bairro certo no momento exato de um surto de CVP.

### Passo 3: Treino do Agente Actor-Critic
Implementação do agente que controlará o gating dinâmico.
- **Arquitetura:** Rede MLP de 3 camadas integrada ao loop de inferência.
- **Comando:** `.venv/Scripts/python.exe tests/Sentinela/train_rl_agent.py`.

### Passo 4: Validação Híbrida (V4 vs V5)
Teste de "Cabo de Guerra":
- Rodar inferência paralela: V4 (Gating fixo) vs V5 (Gating RL).
- **Métrica de Sucesso:** V5 deve superar a V4 em P@10 por pelo menos **5 pontos percentuais**.

---

## 📅 Cronograma Tático
- **Hoje:** Finalização do treino Baseline V4.
- **Amanhã:** Escrita do ambiente de RL e primeiro loop de treino (Warm-up).
- **Final da Semana:** Integração do Agente RL no `app.py`.

> [!IMPORTANT]
> A V5 não descarta a V4; ela a usa como o seu "sistema sensorial", focando toda a capacidade de aprendizado por reforço apenas na **decisão de atenção**.

*Registrado em: 27/04/2026*
