"""
ArchitectureMapper: Mapeia a arquitetura híbrida de Bairros + Nós Dinâmicos.

Define a hierarquia:
  Polígono Bairro (instância gráfica)
    ├─ Centro de Risco (centróide / ponto virtual)
    ├─ Zona de Influência (buffer)
    └─ Vizinhos Imediatos (adjacência geográfica)

Permite:
- Atribuição de ocorrências: Rua → Bairro → Nó
- Propagação de risco: Nó → Bairros Vizinhos
- Visualização híbrida: polígonos + pontos dinâmicos
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class Node:
    """
    Representa um nó no grafo (bairro ou município).
    
    Pode ser:
    - Nó Real: polígono (bairro) com limite definido
    - Nó Dinâmico Virtual: ponto (centróide) sem polígono associado
    """
    
    def __init__(self, idx: int, name: str, node_type: str = 'bairro'):
        self.idx = idx
        self.name = name
        self.node_type = node_type  # 'bairro', 'municipio', 'virtual'
        
        self.geometry = None  # Polygon ou Point
        self.centroid = None  # Point
        self.region = None    # 'fortaleza', 'rmf', 'interior'
        
        # Features associadas
        self.risk_score = 0.0
        self.faction = None
        self.features = {}  # canal 0-25 das features
    
    def set_polygon(self, polygon: Polygon):
        """Define geometria de polígono para nó real."""
        self.geometry = polygon
        self.centroid = polygon.centroid
    
    def set_point(self, point: Point):
        """Define geometria de ponto para nó virtual."""
        self.geometry = point
        self.centroid = point
    
    def is_real(self) -> bool:
        """True se nó tem polígono, False se é apenas ponto."""
        return isinstance(self.geometry, Polygon)
    
    def __repr__(self):
        geom_type = 'polygon' if self.is_real() else 'point'
        return f"Node({self.idx}: {self.name} [{geom_type}] region={self.region})"


class ArchitectureMapper:
    """
    Mapeia a arquitetura híbrida da rede de crime.
    
    Mantém:
    1. Nós Reais (bairros com polígonos) + pontos virtuais (centróides)
    2. Hierarquia de atribuição: ocorrência → bairro → nó
    3. Matriz de adjacência com relações geográficas + territoriais
    """
    
    def __init__(self, nodes_gdf: gpd.GeoDataFrame, num_nodes: int = 319):
        """
        Args:
            nodes_gdf: GeoDataFrame dos nós com estrutura:
                       index, name, geometry (Point ou Polygon), region, node_type
            num_nodes: número total de nós (319)
        """
        self.num_nodes = num_nodes
        self.nodes_gdf = nodes_gdf
        
        # Dicionário de nós indexado
        self.nodes: Dict[int, Node] = {}
        self._initialize_nodes()
        
        # Mapping: bairro_name → node_idx
        self.name_to_idx: Dict[str, int] = {}
        self._build_name_mapping()
        
        # KDTree para atribuição espacial rápida
        self.kdtree = None
        self.kdtree_coords = None
        self._build_kdtree()
        
        # Matriz de adjacência: a ser construída
        self.adjacency_matrix = None
        self.adjacency_type = None  # 'geo', 'faction', 'combined'
        
        logger.info(f"ArchitectureMapper initialized: {num_nodes} nodes, "
                   f"{len(self.nodes)} mapped")
    
    # ========== INITIALIZATION ==========
    
    def _initialize_nodes(self):
        """Cria objetos Node a partir do GeoDataFrame."""
        for idx, row in self.nodes_gdf.iterrows():
            node = Node(idx, row['name'], row.get('node_type', 'bairro'))
            node.geometry = row.geometry
            node.region = row.get('region', 'unknown')
            
            # Extrair centróide
            if hasattr(row.geometry, 'centroid'):
                node.centroid = row.geometry.centroid
            else:
                node.centroid = row.geometry
            
            self.nodes[idx] = node
    
    def _build_name_mapping(self):
        """Cria mapeamento rápido nome → índice (case-insensitive)."""
        for idx, node in self.nodes.items():
            name_normalized = node.name.lower().strip()
            self.name_to_idx[name_normalized] = idx
    
    def _build_kdtree(self):
        """Cria KDTree dos centróides para atribuição rápida de coordenadas."""
        coords = []
        for idx in sorted(self.nodes.keys()):
            node = self.nodes[idx]
            if node.centroid:
                coords.append([node.centroid.x, node.centroid.y])
            else:
                coords.append([0, 0])
        
        if coords:
            self.kdtree_coords = np.array(coords)
            self.kdtree = KDTree(self.kdtree_coords)
            logger.debug(f"KDTree built with {len(coords)} coordinates")
    
    # ========== NODE LOOKUP ==========
    
    def get_node_by_name(self, name: str) -> Optional[Node]:
        """
        Retorna nó pelo nome (normalizado).
        
        Args:
            name: nome do bairro/município
            
        Returns:
            Node ou None se não encontrado
        """
        name_normalized = name.lower().strip()
        idx = self.name_to_idx.get(name_normalized)
        if idx is not None:
            return self.nodes.get(idx)
        return None
    
    def get_node_by_idx(self, idx: int) -> Optional[Node]:
        """Retorna nó pelo índice."""
        return self.nodes.get(idx)
    
    def get_nodes_in_region(self, region: str) -> List[Node]:
        """Retorna todos os nós de uma região."""
        return [n for n in self.nodes.values() if n.region == region]
    
    def get_real_nodes(self) -> List[Node]:
        """Retorna nós com polígonos (nós reais, não virtuais)."""
        return [n for n in self.nodes.values() if n.is_real()]
    
    def get_virtual_nodes(self) -> List[Node]:
        """Retorna nós virtuais (apenas centróides)."""
        return [n for n in self.nodes.values() if not n.is_real()]
    
    # ========== SPATIAL ASSIGNMENT ==========
    
    def assign_occurrence_to_node(self, 
                                  lat: float, 
                                  lng: float,
                                  max_distance_km: float = 5.0) -> Tuple[Optional[int], float]:
        """
        Atribui uma ocorrência (coordenada) ao nó mais próximo.
        
        Estratégia:
        1. Se coordenada está dentro de polígono → nó daquele polígono
        2. Senão, KDTree nearest neighbor até max_distance_km
        
        Args:
            lat, lng: coordenadas da ocorrência
            max_distance_km: máxima distância para atribuição
            
        Returns:
            (node_idx, distance_km) ou (None, inf) se unassigned
        """
        point = Point(lng, lat)
        
        # Técnica 1: Ponto dentro de polígono?
        for idx, node in self.nodes.items():
            if node.is_real() and hasattr(node.geometry, 'contains'):
                if node.geometry.contains(point):
                    return idx, 0.0
        
        # Técnica 2: Vizinhança de borda (buffer)
        for idx, node in self.nodes.items():
            if node.is_real() and hasattr(node.geometry, 'distance'):
                dist_degrees = node.geometry.distance(point)
                dist_km = dist_degrees * 111  # approx
                
                if dist_km < 0.5:  # muito perto da borda
                    return idx, dist_km
        
        # Técnica 3: Nearest neighbor via KDTree
        if self.kdtree is not None:
            dist_degrees, idx_in_tree = self.kdtree.query([lng, lat], k=1)
            dist_km = dist_degrees * 111
            
            if dist_km <= max_distance_km:
                # Encontrar índice real no dicionário
                sorted_indices = sorted(self.nodes.keys())
                node_idx = sorted_indices[idx_in_tree]
                return node_idx, dist_km
        
        return None, float('inf')
    
    def assign_batch_occurrences(self,
                                 occurrences: List[Dict],
                                 location_key: str = 'location') -> np.ndarray:
        """
        Atribui lote de ocorrências em paralelo (via busca espacial).
        
        Args:
            occurrences: lista de {'lat': float, 'lng': float, 'location': str}
            location_key: chave do dict com nome do local (fallback)
            
        Returns:
            array de node indices, -1 se unassigned
        """
        assignments = np.full(len(occurrences), -1, dtype=np.int32)
        
        for i, occ in enumerate(occurrences):
            # Tentar por coordenadas
            if 'lat' in occ and 'lng' in occ:
                node_idx, _ = self.assign_occurrence_to_node(occ['lat'], occ['lng'])
                if node_idx is not None:
                    assignments[i] = node_idx
                    continue
            
            # Fallback: tentar por nome
            if location_key in occ:
                node = self.get_node_by_name(occ[location_key])
                if node:
                    assignments[i] = node.idx
        
        return assignments
    
    # ========== NEIGHBORHOOD ANALYSIS ==========
    
    def get_neighbors(self, node_idx: int, distance_km: float = 2.5) -> Set[int]:
        """
        Retorna índices de nós vizinhos dentro de distância.
        
        Args:
            node_idx: nó central
            distance_km: raio de vizinhança
            
        Returns:
            set de índices vizinhos
        """
        node = self.nodes.get(node_idx)
        if not node:
            return set()
        
        neighbors = set()
        
        for other_idx, other in self.nodes.items():
            if other_idx == node_idx:
                continue
            
            if node.centroid and other.centroid:
                dist_degrees = node.centroid.distance(other.centroid)
                dist_km = dist_degrees * 111
                
                if dist_km <= distance_km:
                    neighbors.add(other_idx)
        
        return neighbors
    
    def get_adjacency_matrix_geo(self) -> np.ndarray:
        """
        Constrói matriz de adjacência baseada em distância geográfica.
        
        A[i,j] = 1 se j é vizinho de i (distância <= 2.5 km)
        
        Returns:
            (num_nodes, num_nodes) binary adjacency matrix
        """
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for node_idx in range(self.num_nodes):
            neighbors = self.get_neighbors(node_idx, distance_km=2.5)
            for neighbor_idx in neighbors:
                adj[node_idx, neighbor_idx] = 1.0
        
        # Simetria: se j vizinho de i, então i vizinho de j
        adj = np.maximum(adj, adj.T)
        
        self.adjacency_matrix = adj
        self.adjacency_type = 'geo'
        
        logger.info(f"Geo adjacency matrix computed: {np.sum(adj)} edges")
        
        return adj
    
    def get_hierarchical_neighborhoods(self) -> Dict[str, List[int]]:
        """
        Retorna hierarquia de vizinhança: Capital → RMF → Interior.
        
        Útil para propagação de risco com limite geográfico/administrativo.
        
        Returns:
            {'fortaleza': [...], 'rmf': [...], 'interior': [...]}
        """
        hierarchy = {
            'fortaleza': [n.idx for n in self.get_nodes_in_region('fortaleza')],
            'rmf': [n.idx for n in self.get_nodes_in_region('rmf')],
            'interior': [n.idx for n in self.get_nodes_in_region('interior')]
        }
        
        return hierarchy
    
    # ========== FEATURE MANAGEMENT ==========
    
    def set_node_features(self, node_idx: int, features: np.ndarray):
        """
        Define features de um nó (26 canais).
        
        Args:
            node_idx: índice do nó
            features: array (26,) com valores [0-100] para CVLI, CVP, etc.
        """
        if node_idx in self.nodes:
            self.nodes[node_idx].features = features
    
    def get_node_features(self, node_idx: int) -> Optional[np.ndarray]:
        """Retorna features de um nó."""
        if node_idx in self.nodes:
            return self.nodes[node_idx].features
        return None
    
    def set_node_risk(self, node_idx: int, risk_score: float):
        """Define score de risco (0-100) para um nó."""
        if node_idx in self.nodes:
            self.nodes[node_idx].risk_score = np.clip(risk_score, 0, 100)
    
    def get_node_risk(self, node_idx: int) -> float:
        """Retorna score de risco de um nó."""
        if node_idx in self.nodes:
            return self.nodes[node_idx].risk_score
        return 0.0
    
    # ========== EXPORT / SERIALIZATION ==========
    
    def export_to_geojson(self, include_risks: bool = True) -> Dict:
        """
        Exporta grafo como GeoJSON com propriedades de risco/features.
        
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for node_idx in sorted(self.nodes.keys()):
            node = self.nodes[node_idx]
            
            # Construir properties
            props = {
                'node_id': node.idx,
                'name': node.name,
                'region': node.region,
                'is_real': node.is_real(),
                'type': node.node_type
            }
            
            if include_risks:
                props['risk_score'] = node.risk_score
            
            # Processar geometria
            if node.is_real():
                geom = node.geometry.__geo_interface__
            else:
                geom = node.centroid.__geo_interface__
            
            feature = {
                'type': 'Feature',
                'geometry': geom,
                'properties': props
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }
    
    def export_topology(self) -> Dict:
        """
        Exporta topologia do grafo (para visualização de redes).
        
        Returns:
            {'nodes': [...], 'edges': [...]}
        """
        nodes_data = []
        for node_idx in sorted(self.nodes.keys()):
            node = self.nodes[node_idx]
            nodes_data.append({
                'id': node.idx,
                'name': node.name,
                'x': node.centroid.x if node.centroid else 0,
                'y': node.centroid.y if node.centroid else 0,
                'region': node.region,
                'risk': node.risk_score
            })
        
        edges_data = []
        if self.adjacency_matrix is not None:
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if self.adjacency_matrix[i, j] > 0:
                        edges_data.append({'source': i, 'target': j})
        
        return {
            'nodes': nodes_data,
            'edges': edges_data
        }
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas da arquitetura."""
        real_nodes = self.get_real_nodes()
        virtual_nodes = self.get_virtual_nodes()
        
        regions = {}
        for region in ['fortaleza', 'rmf', 'interior']:
            regions[region] = len(self.get_nodes_in_region(region))
        
        return {
            'total_nodes': self.num_nodes,
            'real_nodes': len(real_nodes),
            'virtual_nodes': len(virtual_nodes),
            'regions': regions,
            'avg_risk': np.mean([n.risk_score for n in self.nodes.values()]),
            'adjacency_type': self.adjacency_type,
            'adjacency_edges': int(np.sum(self.adjacency_matrix)) if self.adjacency_matrix is not None else 0
        }
