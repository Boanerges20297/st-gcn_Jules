# Nota de Progresso - Tentativa 87 (01/05/2026)

## 🎯 Atividade Atual
Executando o retreino de Fortaleza com o paradigma **ResGAT** e foco em janelas curtas para máxima reatividade tática.

## 🏗️ Mudanças Arquiteturais (O "Tempero")
- **Arquitetura:** `ShallowGAT` -> **ResGAT**. Upgrade para 2 camadas de processamento espaço-temporal com conexão residual.
- **Racional:** Resolver a incapacidade de generalização da camada única. A conexão `out1 + out2` garante que o modelo mantenha o foco no bairro (histórico) enquanto processa o atrito da vizinhança.
- **Ativações:** Substituição total por **PReLU** para melhor fluxo de gradiente em cenários de dados esparsos.

## 📉 Mudanças de Métricas & Hiperparâmetros
- **Configuração de Janela:** Reduzida para **60 dias**. Racional: Eliminar o peso de tendências criminais obsoletas de 2025.
- **Calibração de Loss:** Remoção da regularização L2 sobre as predições. O modelo agora é livre para gerar scores com maior amplitude, facilitando a quebra do platô de ranking.
- **Ranking Weight:** Fixado em **15.0** (Agressivo).
- **Dropout:** Ajustado para **0.4** para sustentar a rede mais profunda.

## 🚀 Status & Próximos Passos
- **Execução:** O script `train_all_specialists.py` está na Época 004.
- **Resultados Parciais (Época 003):** 
    - **P@10:** 35.86%
    - **P@20:** 48.06% (**Sucesso:** Quebra imediata da estagnação).
- **Monitoramento:** Observar a fase de resfriamento do OneCycleLR. O objetivo é buscar o P@10 > 45%.
- **Análise de Potencial:** Decidir a promoção do modelo baseado na curva de convergência linear.
