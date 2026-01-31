Projeto de treino rápido para novo modelo focado em CVLI e facções

Como usar:
- Execute `python train.py` dentro desta pasta (requer PyTorch, numpy, pandas).
- O script carrega `data/processed/graph_data/node_features.npy` e `data/processed/metadata_producao_v2.json`.
- Treina por 5 epochs com janela histórica 120 e horizonte de 30 dias.

Arquivos:
- `train.py`: script de treino e avaliação.
- `config.json`: configurações do experimento.
