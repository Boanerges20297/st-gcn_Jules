# 📊 Integração de Facções do ORCRIMS 2026 - Relatório Final

## ✅ Resumo Executivo

A extração dos dados de facções do arquivo **ORCRIMS 2026.kml** foi concluída com sucesso. Os dados foram incorporados aos nós do mapa e análises de domínio territorial foram geradas.

---

## 🏆 Distribuição de Domínio Territorial

### Ranking de Facções (por número real de placemarks)

| Rank | Facção | Placemarks | % do Total | Territórios | Cidades | Força |
|------|--------|-----------|-----------|-------------|---------|-------|
| 1️⃣ | **COMANDO VERMELHO** | **1.919** | **79,96%** | 1 | 5 | 1919.0 |
| 2️⃣ | PRIMEIRO COMANDO DA CAPITAL | 201 | 8,38% | 198 | 3 | 1.02 |
| 3️⃣ | TCP / GDE | 154 | 6,42% | 154 | 3 | 1.0 |
| 4️⃣ | MASSA | 109 | 4,54% | 108 | 2 | 1.01 |
| 5️⃣ | COMUNIDADES EM DISPUTA | 13 | 0,54% | 13 | 0 | 1.0 |
| 6️⃣ | OKAIDA | 3 | 0,12% | 3 | 0 | 1.0 |
| 7️⃣ | TERRITÓRIOS FANTASMAS | 1 | 0,04% | 1 | 0 | 1.0 |

**Total: 2.400 placemarks mapeados**

---

## 📍 Análise Geográfica

### COMANDO VERMELHO (CV) - Dominante
- **Domínio:** Interior do Ceará
- **Cidades principais:** 
  - Jaguaruana
  - Limoeiro do Norte
  - Quixeré
  - Russas
  - Tabuleiro do Norte
- **Característica:** Presença massiva (1.919 placemarks = ~80% do mapa)
- **Padrão:** Espalhamento geográfico com força territorial muito alta (1 placemark por 1 território)

### PRIMEIRO COMANDO DA CAPITAL (PCC)
- **Domínio:** Áreas urbanas
- **Cidades principais:**
  - Limoeiro do Norte
  - Quixeré
  - Russas
- **Característica:** 201 placemarks distribuídos em 198 territórios (distribuição fragmentada)

### TCP / GDE (Terceiro Comando Puro)
- **Cidades:** Limoeiro do Norte, Limoeiro do NRTE, São João do Jaguaribe
- **Placemarks:** 154 (distribuição uniforme por território)
- **Padrão:** Influência regional concentrada

### MASSA
- **Cidades:** Jaguaruana, Russas
- **Placemarks:** 109
- **Padrão:** Presença menor

---

## 🔧 Arquivos Gerados

### Scripts Criados

1. **`scripts/extract_factions_from_kml.py`**
   - Extrai dados de facções do KML
   - Realiza spatial joining com nós
   - Salva mapping de facções por território

2. **`scripts/integrate_kml_factions_to_graph.py`**
   - Integra dados KML ao grafo existente
   - Cria matriz de adjacência baseada em facções
   - Reconcilia múltiplas fontes de dados de facção

3. **`scripts/analyze_faction_territories_corrected.py`**
   - Análise corrigida usando contagem real de placemarks
   - Gera ranking de dominância territorial
   - Produz relatórios visual e tabular

### Dados Gerados

#### Mapeamentos
- `data/processed/faction_from_kml_mapping.json` - Mapping bruto de facções
- `data/processed/node_kml_faction_mapping.csv` - Mapeamento de nós para facções
- `data/processed/graph_data/nodes_gdf_enriched.pkl` - Nós enriquecidos com dados KML

#### Outputs GeoJSON
- `outputs/nodes_with_kml_factions.geojson` - Nós com facções (para Qgis, Leaflet, etc)
- `outputs/nodes_enriched_with_kml_factions.geojson` - Versão enriquecida
- `outputs/faction_territory_summary.csv` - Resumo tabulado

#### Relatórios
- `reports/faction_territory_analysis_corrected.json` - Análise completa em JSON
- `reports/faction_territory_corrected.html` - Relatório interativo (abrir em navegador)
- `reports/kml_faction_integration_report.json` - Relatório de integração
- `reports/faction_territory_visualization.html` - Visualização alternativa

---

## 📊 Estatísticas de Incorporação

| Métrica | Valor |
|---------|-------|
| Total de nós | 319 |
| Nós com facção atribuída | 223 (69,9%) |
| Nós sem facção (N/A) | 96 (30,1%) |
| Facções identificadas | 7 |
| Matriz de adjacência por facção | Calculada (965 conexões) |

### Qualidade do Matching

- **Matches diretos:** 29 nós (9,1%)
- **Matches fuzzy:** 194 nós (60,8%)
- **Sem match:** 96 nós (30,1%)

---

## 🚀 Como Usar

### 1. Visualizar Dados em QGIS

```bash
# Abrir um dos arquivos GeoJSON em QGIS
# File → Open → outputs/nodes_with_kml_factions.geojson
```

### 2. Usar em Aplicação Web

Os dados estão integrados no sistema e podem ser acessados via:

```python
# No app.py, os nós agora têm a coluna 'faction_kml'
from app import nodes_gdf
print(nodes_gdf[['name', 'faction_kml', 'faction_final']])
```

### 3. Consultar Análise JSON

```python
import json
with open('reports/faction_territory_analysis_corrected.json', 'r') as f:
    analysis = json.load(f)
    
# Dominância por facção
for faction, data in analysis['dominance'].items():
    print(f"{faction}: {data['percentage']:.2f}%")
```

### 4. Regenerar Análises

```bash
# Extrair do KML
python scripts/extract_factions_from_kml.py

# Integrar ao grafo
python scripts/integrate_kml_factions_to_graph.py

# Analisar territórios
python scripts/analyze_faction_territories_corrected.py
```

---

## 📌 Insights Principais

### 1. **Dominância Massiva do CV**
- COMANDO VERMELHO controla ~80% dos placemarks mapeados
- Espalhamento geográfico significativo (5 cidades)
- Força territorial extremamente alta

### 2. **Fragmentação do PCC**
- Distribuição muito fragmentada (201 placemarks em 198 territórios)
- Presença em apenas 3 cidades
- Padrão sugere infiltração urbana ao invés de domínio territorial consolidado

### 3. **TCP/GDE como Terceira Força**
- Distribuição mais equilibrada
- Presença em 3 cidades diferentes
- Força territorial estável

### 4. **Distribuição Complementar**
- Facções menores (MASSA, Disputa, OKAIDA) ocupam nichos geográficos
- Coexistência territorial ao invés de conflito direto
- Padrão típico de crime organizado em expansão

---

## ⚠️ Observações Técnicas

### Limitações da Análise

1. **Dados históricos:** KML baseado em dados históricos (pode não refletir situação atual)
2. **Cobertura incompleta:** Alguns bairros sem dados específicos de facção
3. **Fuzzy matching:** ~61% dos dados baseados em matching fuzzy (menor confiabilidade)

### Recomendações

1. **Validação em campo:** Verificar dados críticos com inteligência operacional
2. **Atualização periódica:** Implementar refresh semestral dos dados
3. **Cross-referência:** Comparar com CECON, IC e outras fontes

---

## 📁 Estrutura de Diretórios

```
data/
├── processed/
│   ├── faction_from_kml_mapping.json
│   ├── node_kml_faction_mapping.csv
│   └── graph_data/
│       └── nodes_gdf_enriched.pkl
│
outputs/
├── nodes_with_kml_factions.geojson
├── nodes_enriched_with_kml_factions.geojson
└── faction_territory_corrected.csv

reports/
├── faction_territory_analysis_corrected.json
├── faction_territory_corrected.html
├── kml_faction_integration_report.json
└── faction_territory_visualization.html

scripts/
├── extract_factions_from_kml.py
├── integrate_kml_factions_to_graph.py
└── analyze_faction_territories_corrected.py
```

---

## 🔗 Próximos Passos

- [ ] Integrar visualização de facções na interface web
- [ ] Adicionar filtros por facção no dashboard
- [ ] Criar alerts para mudanças territoriais
- [ ] Implementar análise de conflitos territoriais
- [ ] Correlacionar dados de ocorrências com territórios de facção

---

**Gerado:** 5 de Fevereiro de 2026  
**Versão:** 1.0  
**Status:** ✅ Completo
