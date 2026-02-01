# Fontes de Polígonos para o Mapa Tático

Para habilitar a visualização de polígonos (limites geográficos) no mapa em vez de apenas pontos, você deve adicionar os seguintes arquivos GeoJSON na pasta `data/raw/`:

## 1. Municípios do Ceará
- **Arquivo esperado:** `data/raw/ceara_municipios.geojson`
- **Conteúdo:** Polígonos de todos os municípios do Ceará.
- **Fonte Sugerida:** IBGE (Malha Municipal) ou GitHub (buscar por "geojson ceara municipios").
- **Propriedades Importantes:** O arquivo deve conter uma propriedade com o nome do município (ex: `name`, `NM_MUN`, `municipio`) para permitir o vínculo com os nós do grafo.

## 2. Bairros de Fortaleza
- **Arquivo esperado:** `data/raw/fortaleza_bairros.geojson`
- **Conteúdo:** Polígonos dos bairros oficiais de Fortaleza.
- **Fonte Sugerida:** Portal de Dados Abertos da Prefeitura de Fortaleza ou GitHub.
- **Propriedades Importantes:** Deve conter o nome do bairro (ex: `NOME`, `bairro`) normalizado.

## 3. Ceará Interior (Opcional/Complementar)
- **Arquivo esperado:** `data/raw/ceara_interior.geojson`
- **Descrição:** Caso utilize uma malha separada para o interior.

## Instruções
1. Baixe os arquivos `.geojson` ou `.json`.
2. Renomeie-os para corresponder exatamente aos nomes acima.
3. Coloque-os na pasta `data/raw/`.
4. Execute o script de processamento:
   ```bash
   python src/data_processing.py
   ```
5. O sistema irá automaticamente mesclar as geometrias aos nós existentes.
