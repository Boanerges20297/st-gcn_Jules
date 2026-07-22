# Diretriz de Dashboard para Zonas GA

## Objetivo operacional

Representar as zonas espaciais otimizadas sem transformar o mapa em uma camada visual confusa.

## Decisao visual

A camada GA deve ser opcional e discreta:
- botao liga/desliga: `Zonas operacionais otimizadas`;
- uma unica cor para todas as zonas;
- preenchimento leve;
- borda solida;
- numero/rank sobre a zona;
- sem gradiente multicolorido;
- sem misturar nova escala de cor com a escala dos bairros.

## Separacao semantica

Mapa de bairros:
- continua comunicando risco geral por territorio administrativo.

Zonas GA:
- comunicam prioridade espacial de campo.

Regra:
- cor = status geral;
- contorno + numero = alvo operacional.

## Fonte de dados inicial

O experimento passou a exportar GeoJSON operacional:

`outputs/experiments/*_latest_ga_zones.geojson`

Cada feature inclui:
- `rank`;
- `bairro`;
- `score`;
- `radius_km`;
- `style_class = ga_operational_zone`.

## Proxima integracao minima

Adicionar no dashboard uma camada Leaflet separada que carregue o GeoJSON mais recente ou um arquivo promovido para caminho estavel.

Estilo recomendado:
- `color: #0f766e`;
- `fillColor: #0f766e`;
- `fillOpacity: 0.12`;
- `weight: 2`;
- rótulo numerico pequeno no centro.

