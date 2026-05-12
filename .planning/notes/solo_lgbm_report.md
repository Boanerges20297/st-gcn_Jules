# 📝 Experimento: LightGBM Solo Flight (Sentinela)
Data: 2026-05-11

## Objetivo
Avaliar o desempenho do LightGBM isoladamente, sem o ensemble de EWMA-Multi que compõe o Sentinela V3.

## Resultados
- **P@10**: 40.0% (Meta V3: 50%)
- **P@20**: 35.0% (Meta V3: 70%)

## Conclusões
1. O LightGBM sozinho captura bem a **Intencionalidade** (cvp_cvli_ratio e target_enc aparecem no topo da importância).
2. No entanto, ele sofre uma degradação severa na **Cobertura Estratégica (P@20)**, caindo de 70% (V3) para 35% (Solo).
3. Isso confirma a validade do **Paradigma Híbrido**: o LGBM foca na precisão tática recente, enquanto o EWMA garante a memória estatística dos hotspots tradicionais.
4. O "Vôo Solo" não é recomendado para produção, mas serve como baseline para provar o valor do Blend.
