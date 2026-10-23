# CHECKPOINT CAN├öNICO ÔÇö CRTF: APRENDIZAGEM DE MUDAN├çA DE REGIME
**Data:** 23/10/2026
**Estado:** planejamento da pr├│xima etapa cient├¡fica ap├│s reconstru├º├úo do CVLI, retreinos ST-GAT/Poisson e avalia├º├úo explorat├│ria de antecipa├º├úo territorial.
**Leia este arquivo PRIMEIRO ao iniciar outro chat.** Em seguida confira os scripts, metadados e a situa├º├úo do Git antes de executar qualquer altera├º├úo.
## 1. Fonte de verdade e ambiente
- **Um ├║nico projeto:** `C:\Users\STI01\Desktop\Projetos\Report Preview`.
- GitHub remoto can├┤nico: `https://github.com/Boanerges20297/st-gcn_Jules.git`; branch **main**, sem PRs. Use o STI atrav├®s do NEXO quando dispon├¡vel.
- GitHub `main` ├® a fonte de verdade; **concilie cuidadosamente** trabalho j├í preparado no STI sem descartar resultados ou reverter logs/cache de processos em execu├º├úo.
- **No instante deste checkpoint:** ├║ltimo commit local/remoto confirmado `22a9d8a7` (retreino ST-GAT/Poisson 2024ÔÇô2026). Existem **17 arquivos de pesquisa preparados no staging do STI, ainda sem commit/push**, incluindo o novo fold de 2023 e o diagn├│stico de ruptura. A publica├º├úo foi barrada por restri├º├Áes de comando no NEXO, **n├úo por falha cient├¡fica**. Verificar novamente o estado real antes de agir.
- N├úo disparar deploy, pipeline de publica├º├úo, PR ou substitui├º├úo dos pesos/modelos operacionais; manter pesquisa experimental isolada. **N├úo alterar OpenVPN, WireGuard ou servi├ºos de rede** durante este trabalho.
- N├úo criar novo reposit├│rio nem separar o CRTF do Report Preview original.
- N├úo usar `git add -A`: os processos do Report Preview modificam caches, logs, dashboards, tensor operacional e sa├¡das continuamente. Selecionar **somente** arquivos relacionados ├á pesquisa e este checkpoint.
## 2. Integridade dos dados e resultados j├í obtidos
- Base original enriquecida CVLI corrigida: `data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`. Foram reconciliadas duplica├º├Áes de representa├º├úo de v├¡timas MySQL, mantendo aliases/IDs e preservando ├│bitos distintos.
- Auditoria: `docs/AUDITORIA_CVLI_MYSQL_SSPDS_20261023.md` e scripts de reparo/testes no `src/core` / `scripts`.
- **Tensor experimental corrigido** de Fortaleza, todos os 121 bairros: `data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl`.
- **SHA-256 congelado da entrada:** `aa32f5fe829ab8345d68e67e2a19523f3886d8740c2f2b392943d29324807d24`. Se o hash mudar, versionar o experimento; **nunca misturar** resultados de fontes diferentes.
- Resultados de retreino ST-GAT v5 do zero e Poisson por ano: `research/benchmarks/cvli_corrected_training_20261023/`. Fold 2023 rec├®m-treinado at├® 31/12/2022 (stride de 7 dias); folds 2024, 2025 e 2026 j├í retreinados. Checkpoints em `exp_models/`.
- Treino original estimava contagem futura de CVLI a 30 dias; **n├úo foi treinado para reconhecer rupturas**. O ST-GAT de contagem n├úo superou consistentemente a refer├¬ncia CVLI-120d para P@10 ao longo dos anos. A resposta **neural pura** e a vers├úo **operacional com suporte hist├│rico** devem ser avaliadas separadamente.
- Compara├º├Áes SSPDS x base: setembro/2026 apresenta diverg├¬ncia (extrato oficial 27 x enriquecido 20), portanto **n├úo interpretar 2026 como valida├º├úo de regimes completa**. Verificar cobertura mensal antes de infer├¬ncia temporal, principalmente em 2026.
## 3. Experimento explorat├│rio de antecipa├º├úo conclu├¡do
- Scripts: `research/benchmarks/crtf_rupture_lead30_20261023.py` e `research/benchmarks/crtf_rupture_lead30_20261023/episode_dedup_diagnostics.py`.
- Relat├│rio detalhado: `research/benchmarks/crtf_rupture_lead30_20261023/RELATORIO_INOVACAO_TERRITORIAL_CRTF.md`.
- Protocolo atual: a partir do momento `t`, tomar hist├│rico territorial de 120 dias, aguardar um intervalo de 30 dias e avaliar dois blocos de 30 dias seguintes (t+30..59 e t+60..89). Rotular crescimento sustentado se cada bloco tiver pelo menos 2 CVLIs e participa├º├úo municipal territorial suavizada pelo menos **35% maior** que a participa├º├úo hist├│rica. A variante retardada exige aus├¬ncia dessa eleva├º├úo nos 30 dias iniciais.
- Os limiares 20%, 35% e 50% foram **explorados retrospectivamente**; n├úo representam defini├º├Áes universais nem prova confirmat├│ria.
- Principais n├║meros (testes 2023ÔÇô2025; 21 datas de emiss├úo): **42 pares bairro-data positivos, mas s├│ 28 epis├│dios distintos** ap├│s agrupar repeti├º├Áes mensais. A primeira oportunidade de aviso Top 10 encontrou **9/28** epis├│dios com a `surpresa neural` (percentil de previs├úo ST-GAT menos percentil do hist├│rico CVLI-120d), contra **6/28** do momentum simples e **2/28** do hist├│rico 120d.
- Para epis├│dios de crescimento retardado foram **18 epis├│dios independentes**, com **7** encontrados pelo residual neural na primeira emiss├úo. Esses n├║meros s├úo explorat├│rios e n├úo demonstram signific├óncia superior aos controles simples.
- Limitador operacional: para o proxy principal, **12 confirma├º├Áes em 210 sele├º├Áes Top 10** nas 21 emiss├Áes (~5,7% das sele├º├Áes). Essas n├úo confirma├º├Áes **n├úo** significam aus├¬ncia de crimes, e sim aus├¬ncia da defini├º├úo de crescimento sustentado; ainda ├® um sistema inadequado para alarmes operacionais.
- Em 2026, nas tr├¬s datas eleg├¡veis com as duas janelas futuras completas antes de setembro, houve **zero r├│tulos positivos** sob essa defini├º├úo. N├úo extrapolar efic├ícia para 2026.
- O resultado mais interessante ├® um **ind├¡cio de inova├º├úo residual antecipat├│ria**, n├úo prova de detectar o nascimento de novos hotspots. Em bairros quase sem atividade hist├│rica o ganho foi fraco. O baseline de momentum simples ├® forte: precisa continuar no protocolo.
## 4. NOVA HIP├ôTESE APROVADA PELO USU├üRIO (23/10/2026)
**Ensinar ao CRTF o pr├│prio significado de uma MUDAN├çA DE REGIME antes de cobrar sua antecipa├º├úo.**
O usu├írio prop├┤s ensinar exemplos visuais/temporais de mudan├ºas, como **+20% ou -20%**, obtidas da forma das curvas de incid├¬ncia. N├úo confundir esse n├║mero ilustrativo com limiar fixo. Uma mudan├ºa de 1 para 2 CVLIs equivale a +100%, mas pode ser ru├¡do; de 10 para 12 (+20%) com persist├¬ncia e contexto pode ser mais consistente. A rede deve entender:
1. **Dire├º├úo:** agrava, reduz/recupera, permanece est├ível.
2. **Magnitude:** mudan├ºa percentual de intensidade/participa├º├úo em rela├º├úo a refer├¬ncia adequada; sa├¡da cont├¡nua com intervalo de incerteza.
3. **Forma temporal:** inclina├º├úo, acelera├º├úo, desacelera├º├úo, salto, retorno, revers├úo, plat├┤, volatilidade, persist├¬ncia.
4. **Tempo de in├¡cio, dura├º├úo e consolida├º├úo:** ponto de mudan├ºa estimado, fase inicial versus regime confirmado.
5. **Contexto:** volume absoluto, vari├óncia estat├¡stica, sazonalidade, tend├¬ncia da cidade; mudan├ºa local n├úo equivale a mudan├ºa do munic├¡pio inteiro.
6. **Confian├ºa:** probabilidade **calibrada em valida├º├úo temporal**, n├úo porcentagem arbitr├íria que parece certeza.
**A mesma metodologia deve detectar tanto AGRAVAMENTO (+%) quanto RECUPERA├ç├âO (-%)**. N├úo assumir que toda alta ├® ruptura nem que toda queda ├® recupera├º├úo definitiva.
Usar a **vis├úo dos dados** como assinatura num├®rica das curvas (features de s├®rie temporal + representa├º├Áes aprendidas), e eventualmente imagens de curvas como apoio did├ítico; n├úo depender de gr├íficos renderizados por computador para ensinar a m├íquina quando os dados originais est├úo dispon├¡veis.
## 5. Pr├│ximos passos em ordem ÔÇö come├ºar pela etapa 0
### Etapa 0 ÔÇö Preserva├º├úo e estado (primeiro comando do pr├│ximo chat)
- Ler este arquivo e `AGENTS.md`, depois os relat├│rios e metadados indicados.
- No STI verificar `git status -sb`, `git diff --cached --name-only`, `git rev-parse HEAD`, `git ls-remote origin refs/heads/main`; verificar os **17 arquivos j├í preparados**. Conciliar com GitHub sem `git add -A` e sem disparar CI/deploy.
- Checar SHA-256, datas completas e volume dos 121 bairros; n├úo usar resultados de setembro/2026 incompletos; preservar v├¡timas m├║ltiplas e aliases MySQL.
- Verificar que todos os experimentos seguintes est├úo **isolados** da produ├º├úo e registram vers├Áes/configura├º├Áes/seeds.
### Etapa 1 ÔÇö Definir o 'vocabul├írio' de regimes com detectores independentes
- Construir prot├│tipo em `research/benchmarks/crtf_regime_learning_20261023/`, **dentro do mesmo projeto**.
- Criar curvas sint├®ticas com r├│tulo conhecido e ru├¡do compat├¡vel com contagens baixas (Poisson/NegBinomial): est├ível, pico isolado, acelera├º├úo gradual, degrau para cima, revers├úo, degrau para baixo, recupera├º├úo sustentada, tend├¬ncia municipal comum.
- Extrair **assinaturas quantitativas** dessas curvas: n├¡vel relativo, varia├º├úo percentual suavizada, derivadas/inclina├º├úo, persist├¬ncia, volatilidade, quantidade absoluta, incerteza e rela├º├úo com a curva municipal. Considerar janelas de **14, 30, 60, 90 e 120 dias** conforme o hist├│rico de experimentos.
- Aplicar a dados reais apenas diagn├│sticos descritivos inicialmente: detector de mudan├ºa de ponto (PELT, CUSUM, BOCPD ou HMM, conforme viabilidade), combinado com limites e persist├¬ncia, sem olhar futuro para emitir o alerta.
- Distinguir `ruido`, `pico`, `transicao_subida`, `novo_regime_alto`, `transicao_descida`, `novo_regime_baixo` e `estabilidade`; ajustar o conjunto de estados se a evid├¬ncia hist├│rica exigir. **N├úo transformar +20% em regra r├¡gida de ruptura.**
- Garantir que os r├│tulos de ensino hist├│ricos retrospectivos s├úo calculados depois dos acontecimentos, mas as vari├íveis dispon├¡veis na emiss├úo **n├úo** usam as janelas futuras. Isso ├® rotulagem supervisionada leg├¡tima, n├úo vazamento.
### Etapa 2 ÔÇö Fazer o baseline 'sem IA'
- Confrontar defini├º├Áes visuais e estat├¡sticas com dados reais, taxa de falsos alarmes e exemplos positivos/negativos.
- Calcular mudan├ºa em rela├º├úo ao *patamar temporal anterior*, incluindo denominadores de participa├º├úo municipal e intensidade absoluta; exibir +20%, -20% e faixas de confian├ºa como significado, n├úo decis├úo determin├¡stica.
- Avaliar detectores estat├¡sticos simples antes do ST-GAT, junto de persist├¬ncia CVLI120, acelera├º├úo 30 vs 90, Poisson e indicador de surpresa neural existente. N├úo escolher limiares depois de ver holdout.
- Gerar um pequeno conjunto audit├ível de exemplos com interpreta├º├úo em portugu├¬s: por que a curva indica ou n├úo uma mudan├ºa de regime.
### Etapa 3 ÔÇö Ensinar o ST-GAT a reconhecer o regime
- Reaproveitar o motor **ST-GAT v5** existente e implementar **cabe├ºa experimental multitarefa**, inicialmente SEM substituir o motor can├┤nico:
  - `magnitude_regime_percentual` (varia├º├úo cont├¡nua esperada, com incerteza);
  - `direcao_regime` (alta, queda, estabilidade);
  - `fase_regime` (ru├¡do/pico/transi├º├úo/consolida├º├úo);
  - `probabilidade_transicao` calibrada;
  - opcional `tempo_ate_transicao` ou hazard para antecipa├º├úo.
- Considerar modelos de duas fases: (a) aprender reconhecimento em curvas passadas/realizadas, (b) aprender antecipa├º├úo a partir do estado observ├ível **at├® t**. **N├úo confundir classificador retrospectivo com previsor**.
- Introduzir custo de falsos alarmes, efeitos da baixa contagem e calibragem de probabilidades; controlar desbalanceamento e estimar intervalos de confian├ºa.
### Etapa 4 ÔÇö Valida├º├úo cient├¡fica/operacional
- Walk-forward estrito; per├¡odos de treino, valida├º├úo e teste separados no tempo, por fold, com agrupamento de epis├│dios para impedir infla├º├úo por meses sobrepostos.
- A ideia 'surpresa neural' e os limiares j├í foram inspecionados em 2023ÔÇô2025; **esses anos n├úo s├úo holdout intocado para confirmar a nova hip├│tese**. Reservar dados realmente novos/prospectivos ou uma avalia├º├úo aninhada com sele├º├úo restrita ├ás dobras de treino, reconhecendo a limita├º├úo.
- Avaliar **anteced├¬ncia em dias, recall de rupturas, precis├úo de alertas, alarmes falsos por bairro/m├¬s, Brier/log-loss, calibra├º├úo, magnitude/dire├º├úo e estabilidade temporal**. Evitar concluir sucesso somente por AUC/P@10/captura de homic├¡dios.
- Comparar os ganhos com modelos estat├¡sticos de regime, baseline sem IA, Poisson, ST-GAT antigo, tend├¬ncia municipal e regra de +20% isolada; testar sensibilidade do denominador e territ├│rios pequenos.
- **N├úo usar a sa├¡da '78% de transi├º├úo' como probabilidade real sem calibra├º├úo verificada.**
### Etapa 5 ÔÇö CRTF din├ómico, somente se houver sinal confi├ível
- Depois de identificar um sinal de transi├º├úo que generaliza, testar algoritmo gen├®tico para ajustar **parti├º├Áes espaciais e raios** em tempo de execu├º├úo, com regulariza├º├úo contra sobreajuste, MAUP e mudan├ºas hist├│ricas de AIS.
- N├úo congelar a geometria sem justificativa; n├úo permitir que GA use r├│tulos futuros do per├¡odo previsto.
- Produzir prova operacional compreens├¡vel: `probabilidade de agravar/diminuir`, `magnitude (+/- %)`, `faixa temporal`, `incerteza` e o significado comparativo para uma decis├úo territorial; sem afirmar causa nem pessoas respons├íveis.
## 6. Defini├º├úo de pronto para o PR├ôXIMO ciclo
O pr├│ximo ciclo ser├í considerado conclu├¡do quando houver (i) **protocolo de r├│tulos/estados versionado**, (ii) **gerador de curvas sint├®ticas e/ou detector estat├¡stico reproduz├¡vel**, (iii) exemplos reais auditados sem vazamento, (iv) compara├º├úo objetiva com baseline simples, (v) testes automatizados de invariantes (pico n├úo virar ruptura, queda n├úo virar alta, baixa contagem n├úo virar alta confian├ºa) e (vi) relat├│rio curto com acertos, erros e pr├│ximos ajustes. S├│ ent├úo passar ao treinamento multitarefa.
## 7. Regras de continuidade
1. **Primeiro vis├úo simples e testes pequenos, depois aprofundar.** Explicar cada f├│rmula relacionada ao problema; reportar achados concretos antes de alegar desempenho.
2. Salvar scripts, outputs e falhas no pr├│prio reposit├│rio; realizar testes e conciliar `main` via STI/GitHub, sem PRs, respeitando permiss├Áes do NEXO.
3. Nenhum CI/deploy de produ├º├úo; n├úo tocar em pesos, APIs ou bancos operacionais.
4. Registre um novo checkpoint (ou atualize este) ap├│s cada experimento relevante. **O pr├│ximo chat deve continuar desta se├º├úo, sem reiniciar a pesquisa CVLI ou refazer retreinos desnecess├írios.**
**Objetivo final:** distinguir *oscila├º├úo*, *nova fase de agravamento* e *nova fase de recupera├º├úo* por assinaturas temporais/espaciais, quantificar a magnitude percentual significativa **em contexto** e testar se a transi├º├úo pode ser antecipada com poucos falsos alarmes.
