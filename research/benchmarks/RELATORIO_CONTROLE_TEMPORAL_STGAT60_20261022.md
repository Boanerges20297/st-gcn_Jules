# Report Preview ÔÇö Controle temporal rigoroso: ST-GAT 60 n├│s / 30 dias
**Execu├º├úo:** 22/10/2026, STI01. **Estado:** pesquisa n├úo promovida, sem altera├º├Áes nos modelos ou nas rotas de produ├º├úo.
## Objetivo
Reavaliar o ganho observado com Top-40 m├│vel e Top-60 fixo, alinhando Poisson e ST-GAT ao horizonte de 30 dias, agora minimizando os sinais potencialmente posteriores ao corte. A investiga├º├úo foi deliberadamente mais r├¡gida que o benchmark de expans├úo anterior.
## Protocolo executado
- Universo de refer├¬ncia: **121 bairros oficiais de Fortaleza**, registrados em `research/benchmarks/exp_data/all_official/processed_fortaleza.pkl`. Vari├ível-alvo: CVLI no futuro, contado com o canal 0 do tensor.
- Malhas: 40 fixos (os originais), 40 m├│veis (recalculados por CVLI nos **730 dias anteriores** ao corte), 60 fixos (40 originais + 20 seguintes no hist├│rico at├® 2025), 60 m├│veis (730 dias anteriores).
- **Poisson 30 dias**, retreinado em 6.540 amostras de datas at├® **30/06/2026**, tendo seus atributos `total_cvli_static`, `recent_cvli_static` e `tension_index` reconstru├¡dos segundo o conhecimento de cada data do treinamento; `tension_index` foi zerado neste controle.
- **ST-GAT v5**, warm start com checkpoint ativo de 40 n├│s (treinado at├® junho/2026); fine-tune nos 60 n├│s (60 janelas de treino; 2023ÔÇôjunho/2026; seis ├®pocas). Usou apenas CVLI historicamente dispon├¡vel, calend├írio e momentum, al├®m de grafo **apenas geogr├ífico**. Intelig├¬ncia/fac├º├Áes/tropa/CVP/clima e pesos de rivalidade do snapshot recente foram deliberadamente exclu├¡dos neste teste.
- **Valida├º├úo para sele├º├úo do checkpoint**: cortes 01/07, 15/07 e 29/07, cujos 30 dias de alvo terminam antes de setembro. **Holdout**: cortes 01/09, 08/09 e 15/09. Controles explorat├│rios adicionais incluem 12 cortes semanais, mas s├úo temporalmente sobrepostos.
- Ambas as sa├¡das t├¬m a mesma interpreta├º├úo operacional: score de ranking; **n├úo s├úo probabilidades calibradas**.
- Captura@10 e Captura@20 t├¬m denominador comum de CVLI nos 121 bairros, nunca apenas o subconjunto observado. P@10 mede propor├º├úo de bairros top-10 com CVLI no pr├│ximo horizonte.
## Achado principal: fine-tuning n├úo validado
No crit├®rio de valida├º├úo de P@10 (30% ST-GAT), o checkpoint original obteve **36,7%** antes do ajuste. Ap├│s seis ├®pocas, o melhor P@10 validado entre os checkpoints efetivamente ajustados foi **30,0%**; portanto, **nenhuma ├®poca ultrapassou o original**.
O seletor de checkpoints devolveu os pesos da **├®poca zero**, sem modifica├º├úo. Isso foi verificado tensor a tensor: `BEST_CKPT_IDENTICAL_TO_ORIGINAL=True`. O arquivo experimental salvo com nome `stgat_v5_60_30d_pit_finetuned_20261022.pth` cont├®m o **checkpoint original selecionado**, n├úo uma vers├úo ajustada superior. **N├úo h├í ganho de retreino demonstrado e n├úo ├® um novo candidato de produ├º├úo.**
## Holdout de setembro ÔÇö tr├¬s cortes de 30 dias (m├®tricas usando os 121 bairros como denominador)
| Malha | Peso ST-GAT / Poisson | P@10 | Captura@20 | Cobertura da malha |
|---|---|---:|---:|---:|
| Top-40 fixo | 20% / 80% | 33,3% | 52,6% | 62,4% |
| Top-40 m├│vel | 20% / 80% | 36,7% | 64,5% | 70,7% |
| Top-40 m├│vel | 40% / 60% | 40,0% | 58,2% | 70,7% |
| Top-60 fixo | 20% / 80% | 36,7% | 64,1% | 97,9% |
| Top-60 fixo | 30% / 70% | 40,0% | 55,8% | 97,9% |
| Top-60 fixo | 40% / 60% | 40,0% | 53,7% | 97,9% |
| Top-60 m├│vel | 20% / 80% | 36,7% | 64,5% | 82,2% |
**Leitura correta:** maior peso no ST-GAT pode melhorar a propor├º├úo de bairros com ocorr├¬ncia entre os dez primeiros, mas **reduz a fra├º├úo dos eventos futuros atingidos pelo Top-20** neste cen├írio de controle; o custo operacional depende de qual m├®trica tem maior valor.
Os tr├¬s cortes de setembro n├úo s├úo independentes ÔÇö horizontes de 30 dias se sobrep├Áem. ├ë uma amostra curta que n├úo fundamenta mudan├ºa de produ├º├úo.
## Compara├º├úo com o benchmark anterior
- A expans├úo para 60 bairros continua ampliando a cobertura potencial. O ganho em cobertura independe de a rede melhorar.
- Os valores de precis├úo do controle estrito **n├úo s├úo diretamente compar├íveis** aos do benchmark anterior como efeito causal de vazamento: al├®m de eliminar sinais de poss├¡vel futuro, removemos vari├íveis leg├¡timas, reconstru├¡mos o grafo e mudamos a janela de treinamento do Poisson.
- No experimento anterior, malha Top-40 m├│vel/30-70 foi candidata promissora. No controle rigoroso, o benef├¡cio ├® **menos consistente** e precisa de mais valida├º├úo.
- Nenhuma propor├º├úo invertida (ST-GAT majorit├írio) ganhou justificativa robusta para produ├º├úo.
## Limita├º├Áes declaradas
1. O ST-GAT de partida foi originalmente treinado com atributos hist├│ricos diferentes; fine-tuning de poucas ├®pocas com features reduzidas n├úo demonstra que uma rede treinada do zero com este novo protocolo seja incapaz de melhorar.
2. A sele├º├úo Top-60 fixa conserva informa├º├úo at├® 2025, o que ├® admiss├¡vel para compara├º├úo de 2026; para treinamento retrospectivo anterior a 2025, a malha fixa ├® retrospectivamente escolhida e pode afetar o aprendizado.
3. O grafo geogr├ífico ├® reconstru├¡do apenas por dist├óncia entre coordenadas atuais, sem os v├¡nculos t├íticos de conflito; ├® uma an├ílise de abla├º├úo, n├úo um substituto imediato.
4. A s├®rie de CVLI pode refletir atrasos de ingest├úo/revis├Áes de classifica├º├úo; seria necess├írio congelar snapshots por data para prova prospectiva estrita.
5. O universo comparado abrange 121 bairros oficiais no arquivo local; dados externos n├úo foram consultados.
## Decis├úo e pr├│ximos passos
**Sem promo├º├úo de modelos ou pesos.** Manter produ├º├úo no Top-40 fixo + h├¡brido 20% ST-GAT/80% Poisson. Preservar os experimentos de Top-40 m├│vel e Top-60 fixo em shadow como hip├│teses.
Para ganhar valor operacional real, priorizar: (a) treino ST-GAT **do zero** com features e adjac├¬ncias reconstru├¡das por corte e treino/val/teste espacial-temporal rigoroso, (b) valida├º├úo prospectiva continuada e (c) indicador que compare **CVLI capturado no Top-10/20** contra custo de deslocamento/alertas. Escolher a propor├º├úo apenas depois dessas verifica├º├Áes.
## Reprodutibilidade
- `stgat60_pit_finetune_20261022.py`: Poisson 30 dias com atributos point-in-time e fine-tune ST-GAT.
- `stgat60_finetune_learning_20261022.csv`: evolu├º├úo de perda e m├®tricas de valida├º├úo, incluindo ├®poca zero.
- `strict30_citydenominator_20261022.py`: compara├º├úo controlada dos quatro esquemas de sele├º├úo territorial.
- `strict30_citydenominator_rows_20261022.csv` e `strict30_citydenominator_summary_20261022.csv`: resultados por corte e por malha.
- `exp_models/` e `exp_data/`: bin├írios **experimentais apenas locais**, isolados e n├úo destinados a commit em Git.
- `stgat60_pit_holdout_metadata_20261022.json`: datas e condi├º├Áes de execu├º├úo.
**Seguran├ºa do projeto:** `src/core/orchestrator.py`, `src/core/data_processing.py`, `models/active/`, `data/processed/` e o processo de publica├º├úo n├úo foram modificados. Arquivos experimentais s├│ em `research/benchmarks/`.
