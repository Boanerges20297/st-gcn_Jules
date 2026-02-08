"""
DynamicAdjacencyManager: Gerencia matriz de adjacência com eventos e fatores temporais.

Implementa 3 tipos de dinamismo:
1. ✅ Event-driven: amplifica vizinhos de áreas com eventos
2. ✅ Temporal: reduz pesos em horas silenciosas  
3. ✅ Decay: influência decai exponencialmente com tempo
"""

import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicAdjacencyMatrix:
    """
    Gerencia matriz de adjacência e a atualiza em tempo real baseado em:
    - Eventos exógenos (conflitos, operações policiais)
    - Padrões temporais (hora do dia, dia da semana)
    - Decaying influence (eventos antigos perdem importância)
    """
    
    def __init__(self, original_adj: np.ndarray, nodes_gdf=None, decay_hours: float = 24.0):
        """
        Args:
            original_adj: (N, N) matriz base de adjacência (geo + faction)
            nodes_gdf: GeoDataFrame com informações dos nós (opcional)
            decay_hours: horas até que influência do evento decay para 50%
        """
        self.original_adj = original_adj.copy()
        self.current_adj = original_adj.copy()
        self.nodes_gdf = nodes_gdf
        self.decay_half_life = decay_hours
        
        # Log de eventos com timestamps
        self.event_log: List[Dict] = []
        
        # Matriz de "intensidade" para cada nó (0-1)
        # Rastreia quão intenso é o dinamismo naquele nó
        self.event_intensity = np.zeros(original_adj.shape[0])
        
        logger.info(f"DynamicAdjacencyManager initialized: {original_adj.shape[0]} nodes")
    
    # ========== EVENT REGISTRATION ==========
    
    def apply_event(self, 
                   event_center_idx: int,
                   severity: str = 'MEDIUM',
                   radius_km: float = 2.0,
                   description: str = '',
                   intensity: float = None) -> Tuple[np.ndarray, List[int]]:
        """
        Aplica efeito de um evento exógeno à matriz de adjacência.
        
        Args:
            event_center_idx: índice do nó afetado
            severity: 'LOW', 'MEDIUM', 'HIGH' - determina amplitude e raio
            radius_km: raio geográfico do efeito (em km)
            description: string descrevendo o evento
            intensity: override do cálculo automático (0-1)
            
        Returns:
            (updated_adj_matrix, affected_node_indices)
        """
        # Map severity to amplification factor
        severity_map = {
            'LOW': {'amp_factor': 1.05, 'radius_mult': 1.0},
            'MEDIUM': {'amp_factor': 1.15, 'radius_mult': 1.5},
            'HIGH': {'amp_factor': 1.30, 'radius_mult': 2.0}
        }
        
        params = severity_map.get(severity, severity_map['MEDIUM'])
        amp_factor = params['amp_factor']
        effective_radius = radius_km * params['radius_mult']
        
        # Se intensity não foi fornecida, calcular a partir da severity
        if intensity is None:
            intensity = {'LOW': 0.3, 'MEDIUM': 0.6, 'HIGH': 0.9}.get(severity, 0.6)
        
        # Encontrar nós próximos
        affected_nodes = self._find_nearby_nodes(event_center_idx, effective_radius)
        
        # Amplificar conexões dos nós afetados
        for node_idx in affected_nodes:
            # Row amplification (outgoing edges)
            self.current_adj[node_idx, :] *= amp_factor
            
            # Column amplification (incoming edges)
            self.current_adj[:, node_idx] *= amp_factor
            
            # Update event intensity tracker
            self.event_intensity[node_idx] = max(self.event_intensity[node_idx], intensity)
        
        # Log do evento
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'center_node': event_center_idx,
            'severity': severity,
            'radius_km': effective_radius,
            'affected_nodes': affected_nodes,
            'description': description,
            'intensity': intensity
        }
        self.event_log.append(event_record)
        
        # Normalizar matriz de adjacência
        self._normalize_adjacency()
        
        logger.info(f"Event applied: {severity} at node {event_center_idx}, "
                   f"affected {len(affected_nodes)} nodes")
        
        return self.current_adj, affected_nodes
    
    def apply_temporal_factors(self, hour: int = None, day_of_week: int = None) -> np.ndarray:
        """
        Aplica multiplicadores temporais baseado em padrões de criminalidade.
        
        Reduz pesos em horas silenciosas (2-6 AM), amplifica em picos.
        
        Args:
            hour: hora do dia (0-23). Se None, usa hora atual
            day_of_week: dia da semana (0=seg, 6=dom). Se None, usa dia atual
            
        Returns:
            updated_adj_matrix com fatores temporais aplicados
        """
        if hour is None:
            hour = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        # Padrões típicos de criminalidade (fatores multiplicativos per hour)
        hourly_factors = {
            0: 0.7,   # 00h (noite, baixo)
            1: 0.6,   # 01h
            2: 0.5,   # 02h (mínimo)
            3: 0.5,   # 03h
            4: 0.6,   # 04h
            5: 0.7,   # 05h
            6: 0.9,   # 06h (amanhecer)
            7: 1.0,   # 07h (subida)
            8: 1.1,   # 08h
            9: 1.2,   # 09h
            10: 1.2,  # 10h
            11: 1.15, # 11h
            12: 1.0,  # 12h (meio-dia)
            13: 1.1,  # 13h
            14: 1.2,  # 14h
            15: 1.2,  # 15h
            16: 1.3,  # 16h (pico)
            17: 1.25, # 17h
            18: 1.1,  # 18h
            19: 1.0,  # 19h
            20: 0.9,  # 20h
            21: 0.9,  # 21h
            22: 0.8,  # 22h (noite)
            23: 0.75  # 23h
        }
        
        # Weekend vs Weekday
        is_weekend = day_of_week >= 5  # sábado = 5, domingo = 6
        weekend_mult = 0.85 if is_weekend else 1.0
        
        # Combinar fatores
        hourly_factor = hourly_factors.get(hour, 1.0)
        combined_factor = hourly_factor * weekend_mult
        
        # Reset from original para aplicar novamente
        self.current_adj = self.original_adj.copy()
        self.current_adj *= combined_factor
        
        # Reapply active events (if any)
        self._reapply_active_events()
        
        logger.debug(f"Temporal factors applied: hour={hour}, "
                    f"day={day_of_week}, factor={combined_factor:.3f}")
        
        return self.current_adj
    
    def apply_decay(self) -> np.ndarray:
        """
        Aplica decaimento exponencial a eventos antigos.
        
        Implementa decay: influence(t) = intensity * exp(-t / decay_half_life)
        
        Returns:
            updated_adj_matrix com eventos decaindo
        """
        now = datetime.now()
        
        # Decay event intensity
        for i, event in enumerate(self.event_log):
            event_time = datetime.fromisoformat(event['timestamp'])
            hours_ago = (now - event_time).total_seconds() / 3600
            
            # Exponential decay
            decay_factor = np.exp(-hours_ago / self.decay_half_life)
            event['active_intensity'] = event['intensity'] * decay_factor
            
            # Remove eventos com intensidade negligenciável (<1%)
            if decay_factor < 0.01:
                self.event_log.pop(i)
        
        # Recalcular matriz a partir dos eventos ativos
        self.current_adj = self.original_adj.copy()
        self._reapply_active_events()
        
        return self.current_adj
    
    # ========== PRIVATE HELPERS ==========
    
    def _find_nearby_nodes(self, center_idx: int, radius_km: float) -> List[int]:
        """
        Encontra nós dentro de raio geográfico do nó central.
        
        Usa GeoDataFrame se disponível, senão usa distância euclidiana simples.
        """
        if self.nodes_gdf is None:
            logger.warning("nodes_gdf not provided, using dummy nearby nodes")
            return [center_idx]
        
        try:
            center_geom = self.nodes_gdf.iloc[center_idx].geometry
            
            # Para polígonos, usar centróide
            if hasattr(center_geom, 'centroid'):
                center_point = center_geom.centroid
            else:
                center_point = center_geom
            
            # Encontrar nós proximos (raio em metros: radius_km * 1000)
            nearby = []
            for idx, row in self.nodes_gdf.iterrows():
                geom = row.geometry
                if hasattr(geom, 'centroid'):
                    point = geom.centroid
                else:
                    point = geom
                
                # Distância aproximada (euclidiana em graus ≈ 111km)
                dist_degrees = center_point.distance(point)
                dist_km = dist_degrees * 111  # conversão simples
                
                if dist_km <= radius_km:
                    nearby.append(idx)
            
            return nearby
        except Exception as e:
            logger.error(f"Error finding nearby nodes: {e}")
            return [center_idx]
    
    def _normalize_adjacency(self):
        """Normaliza matriz usando random walk normalization."""
        # D^-1 @ A (in-degree normalization)
        D_inv = np.diag(1.0 / (np.sum(self.current_adj, axis=0) + 1e-10))
        self.current_adj = D_inv @ self.current_adj
    
    def _reapply_active_events(self):
        """Recalcula efeito cumulativo de todos os eventos ativos."""
        # Reset
        self.current_adj = self.original_adj.copy()
        self.event_intensity = np.zeros(self.original_adj.shape[0])
        
        # Reapply with decay
        now = datetime.now()
        for event in self.event_log:
            event_time = datetime.fromisoformat(event['timestamp'])
            hours_ago = (now - event_time).total_seconds() / 3600
            
            # Exponential decay
            decay_factor = np.exp(-hours_ago / self.decay_half_life)
            if decay_factor < 0.01:
                continue
            
            # Apply with decayed intensity
            intensity = event['intensity'] * decay_factor
            affected_nodes = event['affected_nodes']
            
            amp_factor = 1 + (event.get('amp_factor', 1.15) - 1) * intensity
            
            for node_idx in affected_nodes:
                self.current_adj[node_idx, :] *= amp_factor
                self.current_adj[:, node_idx] *= amp_factor
                self.event_intensity[node_idx] = max(
                    self.event_intensity[node_idx], 
                    intensity
                )
        
        self._normalize_adjacency()
    
    # ========== INTERFACE PÚBLICA ==========
    
    def get_current_matrix(self) -> np.ndarray:
        """Retorna matriz de adjacência atualizada."""
        return self.current_adj.copy()
    
    def get_event_intensity_vector(self) -> np.ndarray:
        """Retorna vetor de intensidade de eventos (0-1 per nó)."""
        return self.event_intensity.copy()
    
    def get_active_events(self) -> List[Dict]:
        """Retorna lista de eventos ativos (com decay > 1%)."""
        now = datetime.now()
        active = []
        
        for event in self.event_log:
            event_time = datetime.fromisoformat(event['timestamp'])
            hours_ago = (now - event_time).total_seconds() / 3600
            decay_factor = np.exp(-hours_ago / self.decay_half_life)
            
            if decay_factor >= 0.01:
                active.append({
                    **event,
                    'hours_ago': hours_ago,
                    'remaining_intensity': event['intensity'] * decay_factor
                })
        
        return active
    
    def clear_events(self):
        """Limpa todos os eventos e reseta para matriz original."""
        self.event_log = []
        self.event_intensity = np.zeros(self.original_adj.shape[0])
        self.current_adj = self.original_adj.copy()
        logger.info("Event log cleared, matrix reset to original")
    
    def export_state(self) -> Dict:
        """Exporta estado atual para serialização."""
        return {
            'timestamp': datetime.now().isoformat(),
            'event_log': self.event_log,
            'event_intensity': self.event_intensity.tolist(),
            'current_adj_shape': self.current_adj.shape,
            'active_events_count': len(self.get_active_events())
        }
    
    def import_state(self, state_dict: Dict):
        """Importa estado anterior (para recovery)."""
        self.event_log = state_dict.get('event_log', [])
        self.event_intensity = np.array(state_dict.get('event_intensity', []))
        self._reapply_active_events()
        logger.info(f"State imported: {len(self.event_log)} events")
