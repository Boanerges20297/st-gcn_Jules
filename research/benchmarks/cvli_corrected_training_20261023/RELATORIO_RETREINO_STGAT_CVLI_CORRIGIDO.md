# ST-GAT v5 ÔÇö retreino apos reconciliacao do CVLI (23/10/2026)
## Status
**Treinos completos** de 2024, 2025 e 2026, com reinicializacao aleatoria do modelo a cada fold. Dados exclusivamente do tensor Fortaleza **121 bairros corrigidos**, proveniente de `data/raw/dados_status_ocorrencias_gerais_ENRIQUECIDO.csv`.
**Aten├º├úo:** nenhum modelo ativo foi alterado e os resultados anteriores sobre CSV com duplicidades nao podem ser comparados diretamente como se usassem o mesmo denominador.
**Tensor:** `data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl` (dentro do proprio `Report Preview`); SHA256 `aa32f5fe829ab8345d68e67e2a19523f3886d8740c2f2b392943d29324807d24`.
## Protocolo
Motor original: `research/benchmarks/rolling_origin_cvli_retrain_20261022.py`, executado pelo novo `research/benchmarks/train_corrected_cvli_20261023.py`. Modelo **DeepSTGAT_v5**, 120 dias de entrada, 41 canais com mascara contra leakage e 30 dias de horizonte. Grafo geodesico; 60 candidatos escolhidos por CVLI dos 730 dias anteriores a cada emissao. Peso inicial aleatorio distinto por fold; **nenhum checkpoint de producao** foi carregado.
- Fold **2024**: treino ate 31/12/2023; validacao janeiro-marco/2024; teste abril-dezembro/2024, 9 janelas.
- Fold **2025**: treino ate 31/12/2024; validacao janeiro-marco/2025; teste abril-dezembro/2025, 9 janelas.
- Fold **2026**: treino ate 31/12/2025; validacao janeiro-marco/2026; teste abril-setembro/2026, 6 janelas (ano parcial com cobertura de fonte que exige verificacao).
Cada fold retreinou tambem **Poisson com alvo CVLI de 30 dias** usando apenas historico permitido pelo corte temporal. Numero de epocas ST-GAT treinadas: 6 (2024, melhor epoca 3); 6 (2025, melhor epoca 4); 5 (2026, melhor epoca 2). Os checkpoints foram recarregados e seus pesos finitos verificados.
## Resultados do holdout
`P@10`: media da proporcao dos 10 bairros escolhidos com ao menos 1 CVLI nos proximos 30 dias.
`Capt20`: media da fracao de CVLI dos 121 bairros capturada entre os 20 escolhidos.
Mesmas janelas e o mesmo universo candidato de 60 bairros em todas as comparacoes.
| Ano | Metodo | P@10 | Capt20 |
|---|---|---:|---:|
| 2024 | CVLI 120 dias | **82,2%** | **49,3%** |
| 2024 | Poisson | 76,7% | 46,1% |
| 2024 | ST-GAT operacional | 80,0% | 48,3% |
| 2024 | ST-GAT saida neural pura | 78,9% | 49,1% |
| 2025 | CVLI 120 dias | 84,4% | 52,4% |
| 2025 | Poisson | 75,6% | 50,5% |
| 2025 | ST-GAT operacional | **85,6%** | 52,4% |
| 2025 | ST-GAT saida neural pura | 84,4% | 52,2% |
| 2026 parcial | CVLI 120 dias | 50,0% | 51,2% |
| 2026 parcial | Poisson | **51,7%** | **56,6%** |
| 2026 parcial | ST-GAT operacional | 46,7% | 52,9% |
| 2026 parcial | ST-GAT saida neural pura | 50,0% | 52,0% |
**Consolidado 2024ÔÇô2025, 18 janelas:** ST-GAT operacional P@10 **82,8%**, Capt20 **50,3%**; CVLI120 dias P@10 **83,3%**, Capt20 **50,9%**; CVLI60 dias Capt20 **52,0%**.
**Consolidado 2024ÔÇô2026 (ano parcial), 24 janelas:** ST-GAT operacional P@10 **73,8%**, Capt20 **51,0%**; CVLI120 dias P@10 **75,0%**, Capt20 **51,0%**; CVLI60 dias Capt20 **52,3%**.
**ST-GAT operacional nao significa saida neural pura**: incorpora suporte do CVLI nos ultimos 30 dias conforme motor historico. A saida neural pura consta separadamente. O hibrido previamente definido 70% ST-GAT/30% Poisson chegou a P@10 **72,1%** nas 24 janelas; nao superou o historico 120 dias.
## Conclusao
O retreino na base corrigida **funcionou e produziu checkpoints utilizaveis**, mas **nao demonstrou generalizacao superior** a uma contagem CVLI de 120 dias para identificar bairros com ocorrencia em 30 dias, nem superioridade de Capt20 frente ao historico 60 dias. A queda em 2026 e consistente com mudanca de regime ou cobertura de dados incompleta; nao afirmar causalidade antes de auditar a fonte de 2026.
Essas comparacoes nao testam ainda a capacidade do CRTF de **antecipar rupturas territoriais**. O proximo experimento cientifico deve definir transicao territorial futura e testar antecipacao com controle temporal, em vez de reotimizar somente ranking historico.
## Reproduzir
Da raiz `C:\\Users\\STI01\\Desktop\\Projetos\\Report Preview`:
```powershell
.\\.venv\\Scripts\\python.exe research/benchmarks/train_corrected_cvli_20261023.py --fold 2024
.\\.venv\\Scripts\\python.exe research/benchmarks/train_corrected_cvli_20261023.py --fold 2025
.\\.venv\\Scripts\\python.exe research/benchmarks/train_corrected_cvli_20261023.py --fold 2026
.\\.venv\\Scripts\\python.exe research/benchmarks/cvli_corrected_training_20261023/compare_against_recency.py
```
Resultados: `research/benchmarks/cvli_corrected_training_20261023/` contendo CSVs de aprendizado, testes, comparacao e metadados. Checkpoints experimentais: `exp_models/stgat_v5_scratch_cvli_corrected_fold_20xx.pth` e `exp_models/poisson_cvli_corrected_fold_20xx.pkl`.
**Sem deploy, PR, troca do modelo ativo ou alteracao do MySQL.**
