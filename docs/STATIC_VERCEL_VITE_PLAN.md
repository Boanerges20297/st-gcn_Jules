# Static Vite Architecture for screenshot-report_preview

## Objetivo

Publicar um mapa operacional estatico na Vercel, atualizado apenas quando um novo snapshot for gerado localmente e enviado ao GitHub.

O produto final deve ser:

- leitura apenas
- sem backend na Vercel
- com metricas congeladas por snapshot
- com atualizacao dependente da atuacao do operador

## Principios

- O projeto atual continua sendo a fonte de verdade operacional.
- O repositorio Vite sera apenas um consumidor de snapshots estaticos.
- Nenhuma metrica critica sera recalculada no browser.
- O frontend da tropa deve operar somente com arquivos JSON e GeoJSON versionados no Git.
- O deploy para Vercel deve ocorrer automaticamente no push, mas a publicacao do snapshot continua manual e controlada.

## Arquitetura

### Camada 1: Sistema principal

Responsabilidades:

- ingestao de eventos
- treino e inferencia
- calculo de momentum
- consolidacao de metricas regionais
- validacao interna
- geracao do snapshot publico

Local: repositorio atual Report Preview.

### Camada 2: Exportador de snapshot

Responsabilidades:

- ler os artefatos ja produzidos pelo sistema principal
- consolidar JSON e GeoJSON para distribuicao estatica
- normalizar nomes, regioes e chaves de lookup
- gerar manifest com data, versao e metadados de publicacao
- copiar os arquivos finais para a estrutura esperada pelo frontend Vite

Script sugerido:

- scripts/export_static_snapshot.py

### Camada 3: Frontend Vite

Responsabilidades:

- carregar o snapshot estatico
- desenhar mapa, filtros e popups
- renderizar top 30 por regiao
- mostrar momentum e metricas congeladas
- exibir metadados do snapshot

Repositorio alvo:

- screenshot-report_preview

### Camada 4: Distribuicao Vercel

Responsabilidades:

- servir o build estatico
- atualizar a cada push no branch configurado
- nao executar regras de negocio

## Fluxo operacional

1. Atualizar dados e rodar pipeline no ambiente interno.
2. Validar o resultado no sistema dinamico atual.
3. Executar o exportador de snapshot.
4. Revisar os artefatos gerados.
5. Copiar os artefatos para o repositorio Vite.
6. Fazer commit e push.
7. Vercel publica o novo estado.

## Escopo do mapa estatico

### Entram no produto estatico

- mapa territorial
- filtros por regiao
- top 30 micronodes por regiao
- camada de micronodes ORCRIM
- faccao predominante
- score de risco congelado
- momentum 7d e 14d
- CVLI recente consolidado
- eventos exogenos consolidados
- ruas criticas
- resumo explicativo congelado
- data de atualizacao do snapshot

### Nao entram no produto estatico

- simulacao
- health dashboard admin
- geocode ao vivo
- parse e save de eventos exogenos
- explainability em runtime via API
- calibracao em runtime
- qualquer POST ou dependencia de backend

## Estrutura do repositorio Vite

```text
screenshot-report_preview/
  public/
    data/
      manifest.json
      dashboard_summary.json
      risk_snapshot.json
      territory_details.json
      explainability.json
      polygons.geojson
      micronodes.geojson
      top30_capital.geojson
      top30_rmf.geojson
      top30_interior.geojson
  src/
    app/
      App.tsx
      router.ts
    components/
      Header.tsx
      RegionFilter.tsx
      MapShell.tsx
      MetricsPanel.tsx
      SnapshotBadge.tsx
      Top30Panel.tsx
      TerritoryDrawer.tsx
    features/
      map/
        leaflet.ts
        layers.ts
        popups.ts
      snapshot/
        loadSnapshot.ts
        selectors.ts
        types.ts
    styles/
      globals.css
      theme.css
    main.tsx
  package.json
  vite.config.ts
  tsconfig.json
```

## Contratos de dados

### 1. manifest.json

Responsavel por informar a versao do snapshot publicado.

```json
{
  "snapshot_id": "2026-03-20T13-21-16",
  "generated_at": "2026-03-20T13:21:16-03:00",
  "source_repo": "Report Preview",
  "source_commit": "abcdef123456",
  "model_label": "ELITE P10",
  "momentum_window_days": 14,
  "regions": ["fortaleza", "rmf", "interior"],
  "notes": "Snapshot estatico publicado manualmente"
}
```

### 2. dashboard_summary.json

Dados leves para cards e cabecalho.

```json
{
  "global": {
    "total_nodes": 109,
    "active_locations": 12,
    "top_region": "rmf"
  },
  "regions": {
    "fortaleza": {
      "avg_risk": 0.31,
      "max_risk": 0.79,
      "top_name": "BARROSO"
    }
  }
}
```

### 3. risk_snapshot.json

Arquivo principal de consulta do frontend.

```json
{
  "items": [
    {
      "id": "fortaleza:barroso",
      "name": "BARROSO",
      "region": "fortaleza",
      "municipality": "Fortaleza",
      "score": 0.79,
      "rank_region": 1,
      "rank_global": 3,
      "momentum_7d": 0.84,
      "momentum_14d": 0.77,
      "recent_cvli": 3,
      "recent_exogenous": 5,
      "faction": "CV",
      "tension_index": 0.91,
      "status": "quente",
      "summary": "Momentum elevado por recorrencia recente e pressao territorial."
    }
  ]
}
```

### 4. territory_details.json

Detalhes usados em popup ou drawer lateral.

```json
{
  "fortaleza:barroso": {
    "critical_streets": [
      "RUA X",
      "RUA Y"
    ],
    "faction": "CV",
    "municipality": "Fortaleza",
    "region": "fortaleza",
    "recent_cvli": 3,
    "recent_exogenous": 5,
    "momentum_7d": 0.84,
    "momentum_14d": 0.77,
    "summary": "Area com pressao persistente no recorte recente."
  }
}
```

### 5. top30_<regiao>.geojson

Camada geometrica pronta para mapa e popup.

Campos minimos esperados em properties:

- node_id
- rank
- name
- municipality
- region
- score
- faction
- momentum_7d
- momentum_14d
- recent_cvli
- recent_exogenous
- is_centroid

### 6. micronodes.geojson

Camada ORCRIM para consulta visual.

Campos minimos esperados em properties:

- micronodo
- area_oficial
- faction
- region
- municipality opcional

## Adaptacao do frontend atual para o frontend Vite

### Endpoints atuais e seus equivalentes estaticos

- /api/risk -> public/data/risk_snapshot.json
- /api/polygons -> public/data/polygons.geojson
- /api/top20_micro_nodes -> public/data/top30_<regiao>.geojson
- /api/micronodes -> public/data/micronodes.geojson
- /api/territory?name= -> public/data/territory_details.json
- /api/explain/<id> -> public/data/explainability.json

### Endpoints que devem ser removidos da versao estatica

- /api/exogenous/parse
- /api/exogenous/save
- /api/simulate
- /api/geocode
- /api/admin/health/*
- /api/model-update-status

## Implementacao sugerida

### Fase 1: Exportador de snapshot

Criar no projeto atual:

- scripts/export_static_snapshot.py

Responsabilidades:

- consolidar outputs/top20_micro_nodes_*.geojson
- consolidar data/raw/inteligencia/micronodos_faccoes_2026.geojson
- consolidar metricas do monitor de eficiencia
- consolidar ranking e dados de momentum
- gerar chaves consistentes por localidade
- salvar em uma pasta local do tipo static_export/

Estrutura sugerida de saida local:

```text
static_export/
  data/
    manifest.json
    dashboard_summary.json
    risk_snapshot.json
    territory_details.json
    explainability.json
    polygons.geojson
    micronodes.geojson
    top30_capital.geojson
    top30_rmf.geojson
    top30_interior.geojson
```

  Comando operacional sugerido:

  ```bash
  python scripts/export_static_snapshot.py
  ```

  Com destino customizado:

  ```bash
  python scripts/export_static_snapshot.py --output-dir static_export/data
  ```

  Saidas geradas pelo exportador:

  - manifest com metadados do snapshot e commit de origem
  - risk_snapshot com ranking consolidado e campos prontos para cards/listas
  - territory_details com detalhe por chave estavel `regiao:nome_normalizado`
  - explainability congelada por node_id quando disponivel
  - polygons, micronodes e top30 por regiao em GeoJSON

### Fase 2: Frontend Vite base

Criar no repositorio screenshot-report_preview:

- Vite
- TypeScript
- Leaflet
- Zustand ou state simples com context, se necessario

Responsabilidades da primeira versao:

- carregar snapshot
- desenhar mapa
- alternar por regiao
- mostrar top 30
- abrir popup com metricas congeladas
- exibir badge de snapshot atualizado em cabecalho

### Fase 3: Refinamento operacional

- drawer lateral de detalhes
- busca local por territorio
- legenda por faccao
- cards de momentum regional
- banner fixo com data da atualizacao

## Processo de publicacao

### Processo manual recomendado

1. Rodar pipeline interno.
2. Validar resultado.
3. Executar exportador de snapshot.
4. Copiar snapshot para screenshot-report_preview/public/data.
5. Rodar preview local do Vite.
6. Fazer commit.
7. Push para GitHub.
8. Vercel publica.

### Convencao de commit sugerida

- chore(snapshot): publish 2026-03-20 13:21
- feat(map): add static territory drawer
- fix(snapshot): normalize faction labels for top30

## Criterios de sucesso

- O mapa abre na Vercel sem backend.
- Todos os popups funcionam com dados locais.
- A camada top 30 muda por regiao com arquivos estaticos.
- A tropa ve momentum e metricas congeladas.
- O snapshot so muda quando um novo push for feito por decisao do operador.

## Recomendacoes finais

- Manter o frontend Vite separado do Flask.
- Nao acoplar o repositorio estatico ao ambiente de treino.
- Evitar logica pesada no browser.
- Usar manifest e timestamp em destaque para reduzir risco operacional.
- Tratar o snapshot publicado como versao aprovada do quadro situacional.