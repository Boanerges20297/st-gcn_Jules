"""
Módulo para logging de predictions do modelo ST-GCN com ranking validation.

Gera arquivos de log detalhados com:
- Ranking atualizado e scores de cada node
- Correções feitas pelo modelo de ranking (demotions)
- Padrões identificados (provenance)
- Estatísticas de distribuição
- Metadados da execução
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class PredictLogger:
    """Logger para predictions do modelo ST-GCN com ranking validation."""
    
    def __init__(self, base_dir: str, nodes_gdf=None):
        """
        Inicializa o logger de predictions.
        
        Args:
            base_dir: Diretório raiz do projeto
            nodes_gdf: GeoDataFrame com informações dos nodes (opcional)
        """
        self.base_dir = base_dir
        self.predicts_dir = os.path.join(base_dir, 'predicts')
        self.nodes_gdf = nodes_gdf
        
        # Garantir que o diretório existe
        os.makedirs(self.predicts_dir, exist_ok=True)
    
    def get_node_name(self, node_id: int) -> str:
        """Obtém o nome do node a partir do node_id."""
        try:
            if self.nodes_gdf is not None and node_id < len(self.nodes_gdf):
                row = self.nodes_gdf.iloc[node_id]
                name = row.get('name') if hasattr(row, 'get') else None
                if name is None:
                    name = row['name'] if 'name' in row else None
                return str(name) if name else f"Area {node_id}"
        except Exception:
            pass
        return f"Area {node_id}"
    
    def generate_timestamp_filename(self) -> str:
        """Gera nome de arquivo com timestamp no formato predict_YYYYMMDD_HHMMSS.txt"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"predict_{timestamp}.txt"
    
    def log_prediction(
        self,
        meta: Dict[str, Any],
        results: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Gera e salva um log detalhado de prediction.
        
        Args:
            meta: Metadados da execução (window, counts, ranking_info, etc)
            results: Lista de resultados por node
            timestamp: Timestamp para o arquivo (default: agora)
            
        Returns:
            Caminho do arquivo gerado
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = self.generate_timestamp_filename()
        filepath = os.path.join(self.predicts_dir, filename)
        
        # Gerar conteúdo do log
        log_content = self._build_log_content(meta, results, timestamp)
        
        # Salvar arquivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        return filepath
    
    def _build_log_content(
        self,
        meta: Dict[str, Any],
        results: List[Dict[str, Any]],
        timestamp: datetime
    ) -> str:
        """Constrói o conteúdo completo do log."""
        
        lines = []
        
        # ==================== HEADER ====================
        lines.append("=" * 80)
        lines.append("ST-GCN PREDICTION LOG - RANKING VALIDATION")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # ==================== 1. RESUMO EXECUTIVO ====================
        lines.append("1. RESUMO EXECUTIVO")
        lines.append("-" * 80)
        lines.extend(self._section_executive_summary(meta, results))
        lines.append("")
        
        # ==================== 2. RANKING ATUALIZADO ====================
        lines.append("2. RANKING ATUALIZADO (TOP 20 NODES)")
        lines.append("-" * 80)
        lines.extend(self._section_ranking(results, meta))
        lines.append("")
        
        # ==================== 3. CORREÇÕES DO MODELO DE RANKING ====================
        lines.append("3. CORREÇÕES FEITAS PELO RANKING (DEMOTIONS)")
        lines.append("-" * 80)
        lines.extend(self._section_ranking_corrections(results))
        lines.append("")
        
        # ==================== 4. PADRÕES IDENTIFICADOS ====================
        lines.append("4. PADRÕES IDENTIFICADOS (PROVENANCE)")
        lines.append("-" * 80)
        lines.extend(self._section_patterns(meta, results))
        lines.append("")
        
        # ==================== 5. DISTRIBUIÇÃO E ESTATÍSTICAS ====================
        lines.append("5. DISTRIBUIÇÃO E ESTATÍSTICAS")
        lines.append("-" * 80)
        lines.extend(self._section_distribution(meta))
        lines.append("")
        
        # ==================== 6. CONFIGURAÇÃO DO MODELO ====================
        lines.append("6. CONFIGURAÇÃO DO MODELO")
        lines.append("-" * 80)
        lines.extend(self._section_configuration(meta))
        lines.append("")
        
        # ==================== 7. EVENTOS EXÓGENOS ====================
        if self._has_exogenous_events(results):
            lines.append("7. EVENTOS EXÓGENOS DETECTADOS")
            lines.append("-" * 80)
            lines.extend(self._section_exogenous_events(results))
            lines.append("")
        
        # ==================== 8. ANÁLISE DETALHADA POR CRÍTICO ====================
        lines.append("8. ANÁLISE CRÍTICA (NODES COM RISCO CRÍTICO)")
        lines.append("-" * 80)
        lines.extend(self._section_critical_analysis(results))
        lines.append("")
        
        # ==================== FOOTER ====================
        lines.append("=" * 80)
        lines.append("FIM DO LOG")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _section_executive_summary(
        self,
        meta: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Seção de resumo executivo."""
        lines = []
        
        # Contagens por severidade
        counts = meta.get('counts', {})
        total = sum(counts.values())
        
        lines.append(f"Total de nodes analisados: {total}")
        lines.append(f"  • Crítico (≥90%):    {counts.get('crítico', 0):>3} ({counts.get('crítico', 0)*100//max(1,total):>3}%)")
        lines.append(f"  • Alto (80-89%):     {counts.get('alto', 0):>3} ({counts.get('alto', 0)*100//max(1,total):>3}%)")
        lines.append(f"  • Moderado (50-79%): {counts.get('moderado', 0):>3} ({counts.get('moderado', 0)*100//max(1,total):>3}%)")
        lines.append(f"  • Baixo (20-49%):    {counts.get('baixo', 0):>3} ({counts.get('baixo', 0)*100//max(1,total):>3}%)")
        lines.append(f"  • Sem Risco (<20%):  {counts.get('sem risco', 0):>3} ({counts.get('sem risco', 0)*100//max(1,total):>3}%)")
        lines.append("")
        
        # Metadados
        lines.append(f"Ranking Source: {meta.get('ranking_source', 'unknown')}")
        lines.append(f"Window CVLI: {meta.get('window_cvli', 'N/A')} dias")
        lines.append(f"Período: {meta.get('window_start', 'N/A')} → {meta.get('window_end', 'N/A')}")
        lines.append(f"Última atualização: {meta.get('last_date', 'N/A')}")
        
        return lines
    
    def _section_ranking(
        self,
        results: List[Dict[str, Any]],
        meta: Dict[str, Any]
    ) -> List[str]:
        """Seção do ranking atualizado (top 20)."""
        lines = []
        
        # Sort by risk_score
        sorted_results = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        lines.append(f"{'Rank':<5} {'Node':<35} {'CVLI%':<7} {'Ranking%':<9} {'Status':<12} {'Provenance':<40}")
        lines.append("-" * 110)
        
        for idx, result in enumerate(sorted_results[:20], 1):
            node_id = result.get('node_id', '?')
            node_name = self.get_node_name(node_id) if isinstance(node_id, int) else str(node_id)
            node_display = f"{node_id} ({node_name})"
            risk_score = result.get('risk_score', 0)
            ranking_score = result.get('ranking_score', None)
            status = result.get('status_label', '?')
            provenance = ', '.join(result.get('score_provenance', [])[:3])
            
            if ranking_score is not None:
                ranking_str = f"{ranking_score:.1f}%"
            else:
                ranking_str = "—"
            
            lines.append(
                f"{idx:<5} {node_display:<35} {risk_score:>6.1f}% {ranking_str:>8} {status:<12} {provenance:<40}"
            )
        
        return lines
    
    def _section_ranking_corrections(self, results: List[Dict[str, Any]]) -> List[str]:
        """Seção de correções feitas pelo ranking (demotions)."""
        lines = []
        
        # Encontrar nodes que foram corrigidos pelo ranking
        demoted = [r for r in results if 'ranking_demoted' in r.get('score_provenance', [])]
        
        if not demoted:
            lines.append("✅ Nenhuma demoção de ranking detectada.")
            lines.append("Todos os nodes mantiveram seu score baseado no modelo ST-GCN.")
            return lines
        
        lines.append(f"🔍 {len(demoted)} nodes corrigidos pelo modelo de ranking:\n")
        
        # Sort by risk_score
        demoted_sorted = sorted(demoted, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        lines.append(f"{'Node':<6} {'CVLI%':<7} {'Ranking%':<9} {'Motivo':<50}")
        lines.append("-" * 80)
        
        for result in demoted_sorted[:15]:
            node_id = result.get('node_id', '?')
            risk_score = result.get('risk_score', 0)
            ranking_score = result.get('ranking_score', 'N/A')
            reasons = result.get('reasons', [])
            motivo = reasons[0][:48] if reasons else "Ranking validation"
            
            lines.append(f"{node_id:<6} {risk_score:>6.1f}% {ranking_score:>8}   {motivo:<50}")
        
        lines.append("")
        lines.append("Interpretar como: O modelo de ranking detectou inconsistência")
        lines.append("e reduziu o score do node para 80% (Alto) para validação adicional.")
        
        return lines
    
    def _section_patterns(
        self,
        meta: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Seção de padrões identificados."""
        lines = []
        
        prov_lists = meta.get('provenance_lists', {})
        
        lines.append("📊 Padrões de risco detectados:\n")
        
        # 1. Histórico
        history_nodes = prov_lists.get('history', [])
        if history_nodes:
            lines.append(f"📍 HISTÓRICO RECENTE ({len(history_nodes)} nodes)")
            lines.append(f"   Nodes com atividade recente no período analisado.")
            top_hist = history_nodes[:5]
            lines.append(f"   Exemplo: {top_hist}")
            lines.append("")
        
        # 2. Muito ativo
        very_active_nodes = prov_lists.get('very_active', [])
        if very_active_nodes:
            lines.append(f"🔴 ALTA ATIVIDADE ({len(very_active_nodes)} nodes)")
            lines.append(f"   Nodes com 3+ homicídios na janela de 14 dias.")
            top_active = very_active_nodes[:5]
            lines.append(f"   Exemplo: {top_active}")
            lines.append("")
        
        # 3. Eventos exógenos
        exo_nodes = prov_lists.get('exogenous', [])
        if exo_nodes:
            lines.append(f"⚠️ EVENTOS EXÓGENOS ({len(exo_nodes)} nodes)")
            lines.append(f"   Nodes afetados por conflitos ou eventos especiais.")
            top_exo = exo_nodes[:5]
            lines.append(f"   Exemplo: {top_exo}")
            lines.append("")
        
        # 4. Eventos exógenos críticos
        exo_crit_nodes = prov_lists.get('exogenous_critical', [])
        if exo_crit_nodes:
            lines.append(f"🚨 EVENTOS CRÍTICOS ({len(exo_crit_nodes)} nodes)")
            lines.append(f"   Nodes com conflitos de alta severidade.")
            top_crit = exo_crit_nodes[:5]
            lines.append(f"   Exemplo: {top_crit}")
            lines.append("")
        
        # 5. Boost de vizinhos
        neighbor_boost_nodes = prov_lists.get('neighbor_boost', [])
        if neighbor_boost_nodes:
            lines.append(f"🗺️ BOOST POR VIZINHANÇA ({len(neighbor_boost_nodes)} nodes)")
            lines.append(f"   Nodes elevados por proximidade com área crítica.")
            top_neigh = neighbor_boost_nodes[:5]
            lines.append(f"   Exemplo: {top_neigh}")
            lines.append("")
        
        # Resumo estatístico
        lines.append("📈 Resumo Estatístico:")
        total_with_pattern = len(set(
            history_nodes + very_active_nodes + exo_nodes + 
            exo_crit_nodes + neighbor_boost_nodes
        ))
        total_nodes = len(results)
        lines.append(f"   Nodes com padrão detectado: {total_with_pattern}/{total_nodes} ({total_with_pattern*100//max(1,total_nodes)}%)")
        
        return lines
    
    def _section_distribution(self, meta: Dict[str, Any]) -> List[str]:
        """Seção de distribuição e estatísticas."""
        lines = []
        
        dist = meta.get('distribution', {})
        history = meta.get('history_stats', {})
        
        if dist:
            lines.append("📊 Distribuição de Scores (Normalizados):")
            lines.append(f"  Mínimo: {dist.get('norm_min', 'N/A'):.1f}%")
            lines.append(f"  Máximo: {dist.get('norm_max', 'N/A'):.1f}%")
            lines.append(f"  Média:  {dist.get('norm_mean', 'N/A'):.1f}%")
            
            percentiles = dist.get('norm_percentiles', {})
            if percentiles:
                lines.append(f"\n  Percentis:")
                for p, v in sorted(percentiles.items()):
                    lines.append(f"    P{p:>2}: {v:>6.1f}%")
            lines.append("")
        
        if history:
            lines.append("📊 Estatísticas de Histórico (CVLI 14 dias):")
            lines.append(f"  Mínimo: {history.get('hist_min', 'N/A')} ocorrências")
            lines.append(f"  Máximo: {history.get('hist_max', 'N/A')} ocorrências")
            lines.append(f"  Média:  {history.get('hist_mean', 'N/A'):.1f} ocorrências")
            
            percentiles = history.get('hist_percentiles', {})
            if percentiles:
                lines.append(f"\n  Percentis:")
                for p, v in sorted(percentiles.items()):
                    lines.append(f"    P{p:>2}: {v:>3} ocorrências")
        
        return lines
    
    def _section_configuration(self, meta: Dict[str, Any]) -> List[str]:
        """Seção de configuração do modelo."""
        lines = []
        
        lines.append(f"Window CVLI: {meta.get('window_cvli', 'N/A')} dias")
        lines.append(f"Window CVP:  {meta.get('window_cvp', 'N/A')} dias")
        lines.append(f"Ranking Source: {meta.get('ranking_source', 'unknown')}")
        
        ranking_info = meta.get('ranking_info', {})
        if ranking_info:
            lines.append("")
            lines.append("Ranking Validation Thresholds:")
            lines.append(f"  Top 1%:  {ranking_info.get('top_1_percent_threshold', 'N/A'):.1f}%")
            lines.append(f"  Top 5%:  {ranking_info.get('top_5_percent_threshold', 'N/A'):.1f}%")
            lines.append(f"  Top 10%: {ranking_info.get('top_10_percent_threshold', 'N/A'):.1f}%")
            lines.append(f"  Método: {ranking_info.get('method', 'unknown')}")
        
        return lines
    
    def _section_exogenous_events(self, results: List[Dict[str, Any]]) -> List[str]:
        """Seção de eventos exógenos."""
        lines = []
        
        # Encontrar nodes com eventos exógenos
        exo_nodes = [r for r in results if 'exogenous' in r.get('score_provenance', []) or 
                     'exogenous_critical' in r.get('score_provenance', [])]
        
        if not exo_nodes:
            lines.append("Nenhum evento exógeno detectado nesta execução.")
            return lines
        
        lines.append(f"Total: {len(exo_nodes)} areas afetadas\n")
        
        for result in exo_nodes[:10]:
            node_id = result.get('node_id', '?')
            node_name = self.get_node_name(node_id) if isinstance(node_id, int) else str(node_id)
            risk_score = result.get('risk_score', 0)
            reasons = result.get('reasons', [])
            is_critical = 'exogenous_critical' in result.get('score_provenance', [])
            
            severity = "🚨 CRÍTICO" if is_critical else "⚠️ MODERADO"
            lines.append(f"Node {node_id} ({node_name}) - {severity} ({risk_score:.1f}%)")
            
            # Mostrar razões
            for reason in reasons[:2]:
                if '🔴' in reason or '⚠️' in reason:
                    lines.append(f"  {reason}")
            lines.append("")
        
        return lines
    
    def _section_critical_analysis(self, results: List[Dict[str, Any]]) -> List[str]:
        """Seção de análise detalhada de nodes críticos."""
        lines = []
        
        # Nodes críticos
        critical = [r for r in results if r.get('risk_score', 0) >= 90]
        
        if not critical:
            lines.append("✅ Nenhum node com risco crítico (≥90%) detectado.")
            return lines
        
        lines.append(f"⚠️ {len(critical)} nodes em nível CRÍTICO:\n")
        
        sorted_critical = sorted(critical, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        for idx, result in enumerate(sorted_critical[:10], 1):
            node_id = result.get('node_id', '?')
            node_name = self.get_node_name(node_id) if isinstance(node_id, int) else str(node_id)
            risk_score = result.get('risk_score', 0)
            cvli_pred = result.get('cvli_pred', 0)
            faction = result.get('faction', 'N/A')
            reasons = result.get('reasons', [])
            
            lines.append(f"{idx}. Node {node_id} ({node_name}) - {risk_score:.1f}% | CVLI Pred: {cvli_pred:.1f} | Facção: {faction}")
            for reason in reasons[:3]:
                lines.append(f"   • {reason}")
            lines.append("")
        
        return lines
    
    def _has_exogenous_events(self, results: List[Dict[str, Any]]) -> bool:
        """Verifica se há eventos exógenos nos resultados."""
        for result in results:
            provenance = result.get('score_provenance', [])
            if 'exogenous' in provenance or 'exogenous_critical' in provenance:
                return True
        return False
