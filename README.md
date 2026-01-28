# st-gcn_jules — Atualização e recuperação após reescrita de histórico

Aviso rápido: o histórico do repositório foi reescrito para remover um arquivo binário grande (`data/processed/processed_graph_data.pkl`). Foi criado um branch de backup remoto chamado `backup-before-filterrepo` e outro `backup-before-strip` antes das mudanças.

Se você clonou o repositório antes desta alteração, siga uma das opções abaixo para sincronizar seu clone:

- Recomendado (re-clonar):

```bash
git clone https://github.com/Boanerges20297/st-gcn_Jules.git
```

- Alternativa (atualizar clone existente — destrutivo localmente):

```bash
# descarta suas mudanças locais e sincroniza com o main reescrito
git fetch origin
git checkout main
git reset --hard origin/main
```

Notas importantes:
- Backup remoto criado: `backup-before-filterrepo` e `backup-before-strip`. Use-os se precisar recuperar estados antigos.
- O arquivo `data/processed/processed_graph_data.pkl` foi removido do histórico e também NOTA: não está mais presente no repositório principal.
- Os dados processados agora estão divididos em `data/processed/graph_data/` (arquivos menores). Use esses arquivos para desenvolvimento local.

Recomendações para colaboradores:
- Reclone se tiver dúvidas.
- Se você manter branches locais que precisa preservar, exporte patches antes de reset:

```bash
git format-patch origin/main..my-branch
```

Sobre arquivos grandes futuros:
- Considere usar Git LFS para blobs grandes. Para mover um arquivo atual para LFS (sem reescrever histórico):

```bash
git lfs install
git lfs track "data/processed/graph_data/*.npy"
git add .gitattributes
git rm --cached data/processed/graph_data/node_features.npy
git add data/processed/graph_data/node_features.npy
git commit -m "Track large data files with Git LFS"
git push origin main
```

Contato:
- Se algo quebrar, me avise antes de seguir passos destrutivos — posso ajudar a recuperar ou criar instruções específicas.
