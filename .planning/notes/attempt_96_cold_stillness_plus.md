# 🧠 Nota de Estratégia: Tentativa 96 - Cold Stillness+

## 🎯 Objetivo
Acelerar a convergência estática mantendo a blindagem contra colapso estrutural, buscando romper o teto de 51%.

## 🛠️ Ajuste de Rota (Pós-Sucesso T95)
A Tentativa 95 provou que o regime de inércia estática é o "porto seguro" para o sistema com Ranking Weight 12.0. Contudo, a velocidade de 0.0001 (causada pelo hardcode corrigido) foi excessivamente lenta.

**Estratégia de Cadência Otimizada (T96):**
1.  **Aumento Cirúrgico (LR 0.0003):** Elevamos a taxa em 3x em relação à rodada anterior, mas mantendo-a em regime **Estático**. Sem a aceleração do OneCycle, o risco de explosão de gradiente continua neutralizado.
2.  **Manutenção do Contexto (Window 60d):** A janela ampliada continua sendo nossa principal defesa contra ruídos, fornecendo uma estimativa de ranking mais estável para o otimizador AdamW.
3.  **Monitoramento de Gradiente:** O alvo é manter o gradiente entre **10.0 e 15.0**. Se cruzar os 20.0 de forma sustentada, a inércia estática pode estar falhando.

## 📈 Expectativa
Uma curva de aprendizado mais íngreme que a da T95, com potencial para atingir o recorde de **54% de P@20** dentro das primeiras 60 épocas.
