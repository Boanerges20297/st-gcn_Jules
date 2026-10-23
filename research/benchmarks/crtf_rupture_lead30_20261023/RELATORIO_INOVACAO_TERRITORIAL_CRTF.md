# CRTF ÔÇö Antecipacao de crescimento territorial: teste com 30 dias de folga
Data: 23/10/2026
**Projeto unico:** `C:\\Users\\STI01\\Desktop\\Projetos\\Report Preview` (`main`).
## Pergunta e metodo
Os checkpoints ST-GAT v5 ja treinados sobre o CVLI corrigido, quando usados sem qualquer novo treinamento sobre a resposta futura, antecipam **aumento sustentado da participacao territorial na violencia letal** melhor que as regras historicas triviais?
No instante `t`, cada territorio tem um historico de 120 dias e seu peso em Fortaleza. O sinal e emitido em `t`. Os primeiros 30 dias sao um intervalo de folga; o evento e verificado em **t+30..59** e **t+60..89**, dois intervalos independentes de 30 dias. Requer **pelo menos 2 CVLIs em cada janela** e aumento da participacao territorial em **>=35%** em *cada* janela relativamente ao historico de 120 dias. Para evitar volatilidade dos territorios de baixa contagem, as participacoes sao suavizadas como `(mortes_do_territorio + 0.5) / (mortes_da_cidade + 0.5*121)`.
A variante **crescimento retardado** exige tambem que a participacao nos primeiros 30 dias ainda esteja **abaixo** do limite de +35%. Isso restringe o teste de antecipacao, mas nao substitui um verdadeiro detector estatistico de mudanca de regime nem comprova sua causa.
O universo espacial permanece em **121 bairros**. A selecao dos **60 candidatos** e feita exclusivamente a partir dos 730 dias anteriores a `t`; sem usar os CVLIs futuros. O ST-GAT e Poisson sao avaliados nos **mesmos 60 candidatos**. Os 30 dias previstos originalmente pelo ST-GAT sao diferentes do alvo de crescimento tardio, aqui avaliado como propriedade indireta (nao treinada para tal).
Foram reexecutados os checkpoints do modelo de cada ano e os **48 rankings de Top 10 em horizonte original de 30 dias (ST-GAT/Poisson) coincidiram com os benchmarks ja salvos**, validando a reconstrucao das previsoes.
## Treinamento temporal e cobertura
- **2023:** ST-GAT v5 treinado somente ate 31/12/2022, com 23 amostras historicas (stride 7 dias pela limitacao de 2022), validacao em janeiro-marco/2023; teste abril-dezembro/2023.
- **2024:** treino ate dezembro/2023; teste abril-dezembro/2024.
- **2025:** treino ate dezembro/2024; teste abril-dezembro/2025.
- **2026 (parcial):** treino ate dezembro/2025; testes de abril-junho cujas verificacoes terminam antes de setembro/2026. Nao usar setembro porque o CSV local contem 20 CVLIs enquanto o extrato SSPDS preservado registra 27.
Para manter as duas janelas de verificacao dentro de cada ano completo, so entram as **7 datas de abril a outubro** em 2023, 2024 e 2025 (21 emiss├Áes). Em 2026 entram tres; **nao houve evento positivo da definicao acima**, logo **nao e valido alegar acuracia de ruptura em 2026**.
## Dados reais x conclusoes
| Indicador | Resultado (2023-2025) |
|---|---:|
| Datas de emissao (testes) | 21 |
| Pares territorio-data com crescimento sustentado (+35%) | 42 |
| Episodios territoriais independentes (agrupando meses consecutivos) | **28** |
| Episodios com crescimento retardado, sem elevacao precoce | **18** |
| Episodios sustentados acessiveis na lista historica de 60 candidatos | 26/28 |
| Episodios retardados acessiveis na lista historica de 60 | 16/18 |
Os 42 pares **nao sao 42 rupturas independentes**. O mesmo bairro pode satisfazer o evento em meses consecutivos; o segundo diagnostico agrupa por ano e bairro quando a lacuna entre emissoes e de ate 40 dias. Para avaliacao de episodios, conta-se **apenas a primeira emissao elegivel**, sem beneficiar metodos que detectam mais tarde.
### Diagnostico Top 10 por emissao, 21 datas
| Metodo | Pares positivos identificados / 42 |
|---|---:|
| Historico CVLI de 120 dias | 4 |
| Poisson de contagem | 5 |
| ST-GAT neural bruto | 7 |
| ST-GAT operacional (com suporte de CVLI recente) | 8 |
| Crescimento recente relativo de participacao (30 vs 90 dias) | 10 |
| Poisson: surpresa sobre historico 120 dias | 8 |
| ST-GAT: surpresa sobre historico 120 dias | **12** |
| ST-GAT operacional: surpresa sobre historico 120 dias | **12** |
**Surpresa da rede** = diferenca entre o percentil do score neural e o percentil dos CVLIs nos ultimos 120 dias, calculados somente sobre os 60 candidatos ja selecionados. A diferenca e alta quando o modelo da prioridade acima do que a contagem passada esperaria. Nao e probabilidade calibrada.
### Episodios unicos ÔÇö somente primeira possibilidade de aviso
| Metodo | Crescimento sustentado (de 28) | Crescimento retardado (de 18) |
|---|---:|---:|
| Historico 120 dias | 2 | 2 |
| Mudanca simples de participacao 30 vs 90 dias | 6 | 3 |
| Poisson: surpresa frente a 120 dias | 6 | 3 |
| ST-GAT operacional | 5 | 5 |
| **ST-GAT neural: surpresa frente a 120 dias** | **9** | **7** |
Isso sugere **informacao antecipatoria residual** nas diferencas entre o modelo e a persistencia historica, nao necessariamente mais eventos previstos nem superioridade estatisticamente demonstrada.
### Limites que impedem promocao operacional
1. **Muitos alertas sem confirmacao do tipo de crescimento:** 21 emissoes x 10 territorios = **210 indicacoes**. Somente 12 dessas indicacoes coincidiram com o *proxy de crescimento sustentado*, isto e, **5,7% de precisao para esse evento especifico** (198 nao o confirmaram). Isso **nao significa** que esses 198 territorios nao tiveram CVLI; muitos podem ter violencia persistente sem crescimento relativo.
2. Para crescimento **retardado**, ST-GAT surpresa acertou **7 de 18 episodios unicos**. Ainda nao implica uma regra de alarme com valor preditivo aceitavel.
3. **Sem evidencias estatisticas de superioridade sobre controles simples.** Em 42 pares, o ST-GAT surpresa teve somente **+2** acertos frente ao momentum simples (12 vs 10); intervalo 95% por reamostragem ingenua de datas: **[-5, +9]**. Inclui zero e, por sobreposicao dos horizontes de 90 dias, esse intervalo pode ser excessivamente otimista. A probabilidade condicional sob ranking aleatorio de 60 candidatos para pelo menos 12 acertos e **p=0,024**, mas e **nao ajustada** a 15 metodos e multiplos r├│tulos explorados. Portanto **nao constitui teste confirmatorio significativo**.
4. A mudanca de limiar tambem muda qual metodo lidera: a +20%, ST-GAT operacional iguala/supera o residual no Top10; a +35% e +50%, o residual neural mostra desempenho relativo melhor.
5. **Territorios quase inativos:** para o proxy `dormant135` (no maximo 1 CVLI nos 120 dias anteriores), houve 10 pares positivos, mas a surpresa neural identificou apenas **2** no Top10, e o ST-GAT operacional nenhum. Nao comprovamos descoberta confiavel de novos hotspots.
6. **2026:** cidade atravessa forte contracao de CVLI, parcialmente verificada em fontes publicas, e a base local nao coincide integralmente com setembro. Nao extrapolar taxas historicas para 2026 como se o processo gerador fosse estacionario.
7. O alvo e um **proxy retrospectivo de participacao territorial** (com suavizacao e limite minimo), nao ruptura confirmada por um teste estatistico independente. Nao inferir causalidade, inicio exato da ruptura ou probabilidade para um bairro especifico.
8. **Selecao das estrategias e metricas foi exploratoria e parcialmente posterior a observacao dos resultados**. Nao houve holdout verdadeiramente intocado para validar a descoberta de "surpresa". Estudo prospectivo deve ser congelado antes das observacoes futuras.
## Linha cientifica recomendada ÔÇö proximo protocolo
**Modelar a inovacao territorial, nao apenas o volume previsto de CVLI.** A hipotese agora tem operacionalizacao concreta: dados os registros anteriores a `t`, a rede deve estimar **quanto a participacao territorial futura excedera a persistencia esperada**. Para treinamento subsequente, usar o desvio futuro ajustado a participacao municipal como alvo continuo e penalizar falsos alarmes; manter simultaneamente baselines de momentum e regressao simples. Depois, avaliar a deteccao de transicao/ruptura em holdout cronologico **que nao participou da selecao dos parametros**. Sera necessario testar precocidade e precisao de alertas, nao apenas ranking Top10. Geometria genetica dinamica so deve ser introduzida depois de isolar esse sinal.
## Reproduzir e localizar tudo no Report Preview
```powershell
# Reconstruir os tensores a partir do CSV corrigido se necessario:
.\\.venv\\Scripts\\python.exe src/core/data_processing.py --fortaleza-node-mode all_official --processed-dir data/runtime/research/cvli_rebuilt_20261022/all_official
# Treino extra de 2023 somente com dados de 2022:
.\\.venv\\Scripts\\python.exe research/benchmarks/train_corrected_cvli_20261023.py --fold 2023
# Reexecutar scores e controles a partir dos checkpoints corrigidos:
.\\.venv\\Scripts\\python.exe research/benchmarks/crtf_rupture_lead30_20261023.py
.\\.venv\\Scripts\\python.exe research/benchmarks/crtf_rupture_lead30_20261023/episode_dedup_diagnostics.py
```
Arquivos: `research/benchmarks/crtf_rupture_lead30_20261023/`; checkpoints ST-GAT e Poisson por ano: `research/benchmarks/cvli_corrected_training_20261023/exp_models/`. Tensor corrigido: `data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl` (SHA256 `aa32f5fe829ab8345d68e67e2a19523f3886d8740c2f2b392943d29324807d24`).
**Somente pesquisa:** nenhum checkpoint de producao alterado, nenhum deploy, nenhuma edicao na base MySQL e nenhum nome de vitima publicado nos resultados.
