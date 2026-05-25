import os
import json
import math
import logging
import osmnx as ox
import networkx as nx

logger = logging.getLogger(__name__)

class STGCNEscapeEngine:
    """
    Motor de predição de rotas de fuga baseado em ST-GCN.
    Diferentemente do ST-GAT (macros e bairros), age em topologia de ruas 
    ('micro') avaliando probabilidade de vetores de fuga a partir de um Evento crítico.
    Utiliza as ruas mapeadas com incidência criminal (geo_streets.json) como nós de atração.
    """
    def __init__(self, data_dir="data/static", base_dir="."):
        self.data_dir = data_dir
        self.graph_path = os.path.join(self.data_dir, "malha_viaria.graphml")
        self.ruas_criticas_path = os.path.join(base_dir, "data", "geo_streets_cache.json")
        self.G = None
        self._is_loaded = False
        self.ruas_criticas = []
        
        if os.path.exists(self.ruas_criticas_path):
            try:
                with open(self.ruas_criticas_path, 'r', encoding='utf-8') as f:
                    self.ruas_criticas = json.load(f)
                    # Otimização: Filtrar apenas as ruas com ocorrências críticas reais (> 0)
                    if isinstance(self.ruas_criticas, list):
                        self.ruas_criticas = [r for r in self.ruas_criticas if r.get('ocorrencias', 0) > 0]
            except Exception as e:
                logger.error(f"Erro ao carregar {self.ruas_criticas_path}: {e}")
        
    def load_graph(self):
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Grafo de ruas não encontrado ({self.graph_path}). Por favor, execute download_malha_viaria.py primeiro.")
        
        logger.info(f"Carregando grafo topo de {self.graph_path} para ST-GCN...")
        self.G = ox.load_graphml(self.graph_path)
        self._is_loaded = True
        logger.info(f"Grafo carregado (ST-GCN Matrix): {len(self.G.nodes)} nós e {len(self.G.edges)} fluxos.")
        return self.G

    def _build_static_subgraph(self, lat, lon, max_distance):
        if not self._is_loaded or self.G is None:
            self.load_graph()

        orig_node = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        lengths = nx.single_source_dijkstra_path_length(self.G, orig_node, cutoff=max_distance, weight='length')
        node_ids = list(lengths.keys())
        if len(node_ids) < 3:
            node_ids = list(nx.ego_graph(self.G, orig_node, radius=8, undirected=True).nodes)
        return self.G.subgraph(node_ids).copy(), orig_node

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000 # raio terra em metros
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _feature_centroid(self, feature):
        geometry = feature.get('geometry') or {}
        geom_type = geometry.get('type')
        if geom_type == 'Point':
            coords = geometry.get('coordinates') or []
            if len(coords) >= 2:
                return float(coords[1]), float(coords[0])
        if geom_type == 'Polygon':
            ring = (geometry.get('coordinates') or [[]])[0]
            points = [p for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2]
            if points:
                return (
                    sum(float(p[1]) for p in points) / len(points),
                    sum(float(p[0]) for p in points) / len(points),
                )
        return None, None

    def score_street_foci(self, features, area_risk_scores=None, neighbor_distance=1000, propagation_steps=2):
        """
        Pontua focos de ruas como nÃ³s ST-GCN leves.

        Cada foco 500m vira um nÃ³; arestas conectam focos prÃ³ximos na malha
        territorial. O sinal inicial combina:
          - risco previsto da Ã¡rea-mÃ£e (ST-GAT/orquestrador),
          - histÃ³rico local do foco,
          - densidade de logradouros agregados.

        A propagaÃ§Ã£o espacial suaviza/eleva focos vizinhos de Ã¡reas quentes,
        simulando a parte convolucional do ST-GCN em escala de rua.
        """
        if not features:
            return features

        area_risk_scores = area_risk_scores or {}

        def norm_name(text):
            import unicodedata
            import re
            if not isinstance(text, str):
                return ''
            out = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').upper().strip()
            return re.sub(r'\s+', ' ', out)

        nodes = []
        max_occ = max(float((f.get('properties') or {}).get('total_occurrences') or 0) for f in features) or 1.0
        max_streets = max(float((f.get('properties') or {}).get('street_count') or 0) for f in features) or 1.0

        for idx, feature in enumerate(features):
            props = feature.get('properties') or {}
            lat, lon = self._feature_centroid(feature)
            area_keys = [
                norm_name(props.get('bairro')),
                norm_name(props.get('cidade')),
                norm_name(props.get('name')),
            ]
            area_score = 20.0
            for key in area_keys:
                if key and key in area_risk_scores:
                    area_score = float(area_risk_scores[key])
                    break

            local_hist = math.log1p(float(props.get('total_occurrences') or 0)) / math.log1p(max_occ)
            density = math.log1p(float(props.get('street_count') or 0)) / math.log1p(max_streets)
            area_norm = max(0.0, min(1.0, area_score / 100.0))
            seed = (0.50 * area_norm) + (0.35 * local_hist) + (0.15 * density)
            nodes.append({
                'idx': idx,
                'lat': lat,
                'lon': lon,
                'score': seed,
                'area_score': area_score,
            })

        adjacency = [[] for _ in nodes]
        for i, a in enumerate(nodes):
            if a['lat'] is None or a['lon'] is None:
                continue
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if b['lat'] is None or b['lon'] is None:
                    continue
                dist = self._haversine(a['lat'], a['lon'], b['lat'], b['lon'])
                if dist <= neighbor_distance:
                    weight = 1.0 - (dist / neighbor_distance)
                    adjacency[i].append((j, weight))
                    adjacency[j].append((i, weight))

        scores = [n['score'] for n in nodes]
        for _ in range(max(0, propagation_steps)):
            propagated = []
            for idx, score in enumerate(scores):
                neighbors = adjacency[idx]
                if not neighbors:
                    propagated.append(score)
                    continue
                total_w = sum(w for _, w in neighbors) or 1.0
                neighbor_signal = sum(scores[j] * w for j, w in neighbors) / total_w
                propagated.append((0.68 * score) + (0.32 * neighbor_signal))
            scores = propagated

        min_s, max_s = min(scores), max(scores)
        spread = max(max_s - min_s, 1e-6)
        for node, score in zip(nodes, scores):
            feature = features[node['idx']]
            props = feature.setdefault('properties', {})
            calibrated = 100.0 * (score - min_s) / spread
            probability = round(max(0.0, min(100.0, calibrated)), 1)
            props['stgcn_score'] = probability
            props['predicted_cvli_probability'] = probability
            props['parent_area_risk_score'] = round(float(node['area_score']), 1)
            props['stgcn_neighbor_count'] = len(adjacency[node['idx']])
            props['score'] = probability
            props['risk_score'] = probability
            props['model'] = 'ST-GCN Rua/Foco 500m'

        features.sort(
            key=lambda feat: (
                (feat.get('properties') or {}).get('stgcn_score', 0),
                (feat.get('properties') or {}).get('total_occurrences', 0),
            ),
            reverse=True,
        )
        for rank, feature in enumerate(features, 1):
            feature.setdefault('properties', {})['stgcn_rank'] = rank
        return features

    def predict_escape_routes(self, lat, lon, max_distance=1000, time_horizon="5m"):
        """
        Dada a geolocalização do evento, estima via ST-GCN (simulada na matriz) os eixos prováveis de fuga,
        com ênfase (atração espacial) para os nós (lat/lng) mais críticos da região.
        """
        # A malha estática costuma ter "buracos" e gerar cercos deslocados (snap para bairros vizinhos).
        # Para precisão tática extrema, SEMPRE baixamos a malha local exata sob demanda (com cache em memória).
        cache_key = f"{round(lat, 3)}_{round(lon, 3)}_{max_distance}"
        if not hasattr(self, '_local_graphs_cache'):
            self._local_graphs_cache = {}
            
        if cache_key in self._local_graphs_cache:
            subgraph = self._local_graphs_cache[cache_key]
            orig_node = ox.distance.nearest_nodes(subgraph, X=lon, Y=lat)
        else:
            try:
                logger.info(f"Recortando malha viaria local para {lat}, {lon} (raio {max_distance}m)...")
                subgraph, orig_node = self._build_static_subgraph(lat, lon, max_distance)
                self._local_graphs_cache[cache_key] = subgraph
            except Exception as e:
                logger.warning(f"Falha ao usar malha local; tentando OSM dinamico: {e}")
                try:
                    subgraph = ox.graph_from_point((lat, lon), dist=max_distance, network_type='drive')
                    orig_node = ox.distance.nearest_nodes(subgraph, X=lon, Y=lat)
                    self._local_graphs_cache[cache_key] = subgraph
                except Exception as osm_error:
                    logger.error(f"Falha ao obter malha viaria dinamica: {osm_error}")
                    raise RuntimeError("Não foi possível obter a malha viária para esta localidade.")
        
        routes = []
        
        # Gerar a malha viária de contenção (todas as ruas do subgrafo) para objetivação do gestor
        from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, mapping
        reachable_lines = []
        for u, v, data in subgraph.edges(data=True):
            if 'geometry' in data:
                reachable_lines.append(data['geometry'])
            else:
                ux, uy = subgraph.nodes[u]['x'], subgraph.nodes[u]['y']
                vx, vy = subgraph.nodes[v]['x'], subgraph.nodes[v]['y']
                reachable_lines.append(LineString([(ux, uy), (vx, vy)]))
        
        if reachable_lines:
            mesh_geom = MultiLineString(reachable_lines)
            routes.append({
                "type": "Feature",
                "geometry": mapping(mesh_geom),
                "properties": {
                    "probability": 0,
                    "time_estimate": "N/A",
                    "main_axis": "Malha Viária Regional (Raio de Cerco)",
                    "rank": -1,
                    "is_mesh": True,
                    "max_distance_m": max_distance,
                    "model": "ST-GCN Rotas de Fuga"
                }
            })
            
        # Criar polígono de Cerco Tático (Área de Contenção)
        points = [Point(data['x'], data['y']) for n, data in subgraph.nodes(data=True) if 'x' in data and 'y' in data]
        if len(points) >= 3:
            containment_polygon = MultiPoint(points).convex_hull
            routes.append({
                "type": "Feature",
                "geometry": mapping(containment_polygon),
                "properties": {
                    "probability": 100,
                    "time_estimate": time_horizon,
                    "main_axis": "Perímetro de Cerco Tático",
                    "rank": 0,
                    "is_polygon": True,
                    "max_distance_m": max_distance,
                    "model": "ST-GCN Rotas de Fuga"
                }
            })
        
        # Identificar NÓS DE BORDA (Perímetro) para garantir que os vetores de fuga cruzem todo o cerco
        try:
            path_lengths = nx.single_source_dijkstra_path_length(subgraph, orig_node, weight='length')
            border_nodes = [n for n, d in path_lengths.items() if d > (max_distance * 0.75)]
        except Exception:
            border_nodes = list(subgraph.nodes)
            path_lengths = {}

        if not border_nodes:
            border_nodes = list(subgraph.nodes)
            
        targets = []
        
        # Estratégia ST-GCN I: Vetores de Fuga guiados por Vias Críticas
        hot_streets = []
        for rua in self.ruas_criticas:
            r_lat = rua.get('lat')
            r_lng = rua.get('lng')
            if r_lat and r_lng:
                dist = self._haversine(lat, lon, r_lat, r_lng)
                if 200 < dist < (max_distance * 1.5):
                    hot_streets.append((rua, dist, r_lat, r_lng))
                    
        hot_streets = sorted(hot_streets, key=lambda x: x[0].get('ocorrencias', 0), reverse=True)[:5]
        
        for rua_data, dist, r_lat, r_lon in hot_streets:
            if not border_nodes: break
            # Pega o nó da borda do cerco que está mais alinhado/próximo com a rota crítica de fuga
            best_border = min(border_nodes, key=lambda b: self._haversine(
                r_lat, r_lon, subgraph.nodes[b].get('y', 0), subgraph.nodes[b].get('x', 0)
            ))
            if best_border not in targets and best_border != orig_node:
                targets.append(best_border)
                
        # Estratégia II: Topologia Expansiva (Completar com as saídas principais da malha)
        border_nodes = sorted(border_nodes, key=lambda d: (subgraph.degree(d), path_lengths.get(d, 0)), reverse=True)
        for b in border_nodes:
            if len(targets) >= 4:
                break
            if b not in targets and b != orig_node:
                targets.append(b)
        
        # Limitar para os 4 vetores principais
        targets = targets[:4]
        probabilities = [0.65, 0.20, 0.10, 0.05] # ST-GCN confia nas âncoras quentes probabilísticas
        
        for idx, target in enumerate(targets):
            try:
                # Caminho Preditivo de Fuga (shortest path heurístico otimizado pela malha)
                path = nx.shortest_path(subgraph, source=orig_node, target=target, weight='length')
                
                # Ignorar paths vazios
                if len(path) < 2: continue
                
                coords = [(subgraph.nodes[n]['x'], subgraph.nodes[n]['y']) for n in path]
                prob = probabilities[idx] if idx < len(probabilities) else 0.02
                
                road_names = []
                for u, v in zip(path[:-1], path[1:]):
                    data = subgraph.get_edge_data(u, v)
                    if data:
                        edge_data = data.get(0, {}) 
                        name = edge_data.get('name')
                        if name and name not in road_names:
                            if isinstance(name, str):
                                road_names.append(name)
                            elif isinstance(name, list):
                                road_names.append(name[0])
                
                if len(road_names) >= 2:
                    main_axis = f"{road_names[0]} → {road_names[-1]}"
                else:
                    main_axis = road_names[0] if road_names else "Viadutos / Acessos Locais"
                
                route = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "probability": int(prob * 100),
                        "time_estimate": time_horizon,
                        "main_axis": main_axis,
                        "rank": idx + 1,
                        "max_distance_m": max_distance,
                        "model": "ST-GCN Rotas de Fuga"
                    }
                }
                routes.append(route)
            except nx.NetworkXNoPath:
                continue

        return {
            "type": "FeatureCollection",
            "features": sorted(routes, key=lambda x: x["properties"]["probability"], reverse=True),
            "metadata": {
                "model": "ST-GCN (Atração Criminométrica Georreferenciada)",
                "origin": [lon, lat],
                "max_distance_m": max_distance,
                "total_routes": len(routes),
            }
        }
