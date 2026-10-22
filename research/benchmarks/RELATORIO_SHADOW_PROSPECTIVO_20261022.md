# Report Preview ORIGINAL ÔÇö Protocolo prospectivo selado (2026-10-22)
## Finalidade
Pr├®-registrar uma compara├º├úo **prospectiva** de rankings de risco de CVLI por bairro, no horizonte de 30 dias, sem selecionar resultados ou ajustar pesos depois de observar ocorr├¬ncias futuras. O uso ├® **exclusivamente de pesquisa**; n├úo h├í indica├º├úo de preven├º├úo de CVLI, altera├º├úo de aloca├º├úo operacional ou probabilidades calibradas.
## Cortes fixados antes dos resultados
- Data da ├║ltima observa├º├úo dispon├¡vel: **2026-10-21**.
- Emiss├úo das prioridades: **2026-10-22**.
- Alvo futuro: **2026-10-22 a 2026-11-20**, inclusive.
- Data m├¡nima para avalia├º├úo: **2026-11-27**, condicionado ├á exist├¬ncia de registros completos dos 121 bairros at├® essa data (lag m├¡nimo de sete dias n├úo garante aus├¬ncia de revis├Áes).
- Universo que definir├í o denominador: **121 bairros oficiais de Fortaleza**.
- Candidatos: 40 bairros hist├│ricos originais e 60 (40 originais mais 20 escolhidos usando **somente 2024 e 2025**).
## Cen├írios e m├®tricas pr├®-definidos
Cen├írios A: ST-GAT original nos 40 bairros; A60: transfer├¬ncia zero-shot para 60; B: ST-GAT 60 com fine-tuning (melhor ├®poca 7); C: ST-GAT 60 iniciado do zero (melhor ├®poca 1). O Poisson experimental ├® o mesmo modelo de 30 dias para todos.
Pesos prim├írios fixados de ST-GAT: **20%, 40%, 70%**, combinados com Poisson por ranking percentil. Controles diagn├│sticos: **0%** e **100%** de ST-GAT. Ser├úo relatados P@10 (bairros com CVLI), captura de ocorr├¬ncias no Top-10 e no Top-20 com denominador de 121 bairros, cobertura da malha e falsos positivos no Top-10. **Nenhuma propor├º├úo ser├í otimizada retroativamente** ap├│s o teste.
## Compromisso criptogr├ífico p├║blico, sem listas de bairros futuros
O CSV cont├®m **1.100 linhas de ranking**, todas produzidas antes de se conhecerem os r├│tulos do futuro. Para n├úo expor prioridades prospectivas num reposit├│rio **p├║blico**, o Git abriga somente scripts, protocolo e **digests**, mas n├úo cont├®m o CSV de previs├Áes, os checkpoints nem o dataset selado.
- SHA-256 do CSV de rankings: `7aad6e8baac1416c8ae0dadf85ccf6eb177ef19e6b726ab5ba1eb0df21170494`
- SHA-256 do manifesto de execu├º├úo: `5a5f462f16a3d092c25badd0d6e7329848009c144005691efb6dd91d36fe9fa4`
A previs├úo selada e o manifesto est├úo em armazenamento privado no STI, dentro de `research/benchmarks/shadow_20261022/`, com c├│pia conferida por SHA-256 em `G:\Meu Drive\ChatGPT\Report Preview\shadow_20261022\`. O arquivo de dados e os checkpoints congelados est├úo em `research/benchmarks/exp_data/shadow_20261022/` no STI, com c├│pia no mesmo espelho privado. A exclus├úo dos diret├│rios de previs├Áes e dos bin├írios ├® garantida por `research/benchmarks/.gitignore`.
## Integridade verificada
- CVLI observado em todos os 40 n├│s bate **exatamente** entre a fonte experimental 121 e os dados ativos, por bairro e por data at├® 2026-10-21.
- Os 20 n├│s adicionais conferem com o ranking somente de CVLI 2024ÔÇô2025 e com a ordem armazenada nos dois checkpoints.
- Seis fontes congeladas conferidas por SHA-256 na origem e no arquivo local.
- Nove arquivos do shadow (incluindo rankings e manifesto) verificados por SHA-256 no espelho privado do Google Drive.
- `shadow_cvli_30d_evaluate_20261022.py --integrity-only` passou; tentativa normal de avalia├º├úo antes de 2026-11-27 retornou `NOT_READY_CLOCK` sem escrever resultado.
- C├│digo do projeto em produ├º├úo, checkpoints ativos, tabelas e tela operacional n├úo foram modificados.
## Execu├º├úo futura, ap├│s matura├º├úo
Atualizar a base **completa dos 121 bairros** (n├úo usar somente o `processed_fortaleza.pkl` ativo Top-40), preservando nomes e datas, e executar:
```powershell
.\.venv\Scripts\python.exe research\benchmarks\shadow_cvli_30d_evaluate_20261022.py --all121-processed "CAMINHO_PARA_121_BAIRROS_ATUALIZADOS.pkl"
```
O avaliador verifica o hash do ranking, o vencimento temporal e a presen├ºa de todos os 30 dias e dos 121 bairros. Somente ent├úo grava resultados em arquivo novo sem sobrescrever a previs├úo selada. Repeti├º├Áes devem explicitar revis├Áes dos dados e seus hashes.
## Interpreta├º├úo e limites
Uma ├║nica janela prospectiva de 30 dias oferece evid├¬ncia preliminar e **n├úo demonstra robustez estat├¡stica**. Os atributos t├íticos no snapshot s├úo conhecidos na emiss├úo, mas o hist├│rico de treino n├úo disp├Áe de snapshots passados completamente reconstru├¡dos. A compara├º├úo B/C continua sujeita a sele├º├úo retrospectiva de 60 bairros. Avaliar por ciclos futuros adicionais antes de promover qualquer checkpoint ou altera├º├úo na propor├º├úo 80/20 de produ├º├úo.
