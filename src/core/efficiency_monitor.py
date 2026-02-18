import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.core.orchestrator import normalize_name

class EfficiencyMonitor:
    """
    Monitor de Eficiência Report Preview: Avalia o poder preditivo global do modelo.
    Compara o ranking consolidado de todas as 299 localidades monitoradas com os eventos reais.
    """
    def __init__(self, project_root, orchestrator, nodes_gdf):
        self.root = project_root
        self.orchestrator = orchestrator
        self.nodes_gdf = nodes_gdf
        self.history_path = os.path.join(self.root, "logs", "efficiency_history.json")
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

    def run_evaluation(self):
        """
        Executa a avaliação de eficiência regionalizada (P5, P10, P20).
        Compara o ranking de cada região com os eventos reais dos últimos 7 dias.
        """
        if self.orchestrator is None:
            return None

        try:
            # 1. Obter Predições Consolidadas (Baseline Global)
            scores_map = self.orchestrator.get_combined_risk(None)
            
            # 2. Obter Ground Truth (Eventos dos últimos 7 dias)
            events_path = os.path.join(self.root, "data", "exogenous_events.json")
            if not os.path.exists(events_path):
                return None

            with open(events_path, 'r', encoding='utf-8') as f:
                events = json.load(f)

            today = datetime.now().date()
            window_start = today - timedelta(days=7)
            
            ground_truth = {}
            for e in events:
                try:
                    dstr = e.get('date') or e.get('event_date')
                    e_date = datetime.strptime(dstr[:10], '%Y-%m-%d').date()
                    if e_date >= window_start:
                        loc_raw = e.get('bairro') or e.get('location') or e.get('municipio')
                        if loc_raw:
                            loc_norm = normalize_name(str(loc_raw))
                            ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + 1
                except: continue

            # 3. Preparar Grupos de Avaliação (Global + Regionais)
            regions = {'global': list(scores_map.keys())}
            for r_name, spec in self.orchestrator.specialists.items():
                regions[r_name] = [normalize_name(n) for n in spec['data']['nodes_gdf']['name']]
            
            results = {"date": str(today)}
            
            # 4. Avaliar cada Região
            for r_name, node_list in regions.items():
                # Filtrar scores e ground truth para esta região
                r_scores = {n: scores_map[n] for n in node_list if n in scores_map}
                r_ground_truth = {n: count for n, count in ground_truth.items() if n in node_list}
                
                if not r_scores or not r_ground_truth:
                    results[r_name] = {"status": "no_events", "p5": 0, "p10": 0, "p20": 0}
                    continue
                    
                r_ranking = sorted(r_scores.items(), key=lambda x: x[1], reverse=True)
                
                region_metrics = {
                    "total_nodes": len(r_scores),
                    "active_locations": len(r_ground_truth),
                    "total_events": sum(r_ground_truth.values())
                }
                
                for k in [5, 10, 20]:
                    top_k = [name for name, score in r_ranking[:k]]
                    hits = [name for name in top_k if name in r_ground_truth]
                    precision = len(hits) / k
                    region_metrics[f"p{k}"] = round(precision, 4)
                    region_metrics[f"hits{k}"] = hits  # Dados usáveis: quais bairros acertamos
                
                results[r_name] = region_metrics

            # 5. Salvar Histórico e Retornar
            self.save_to_history(results)
            return results

        except Exception as e:
            print(f"⚠️ Erro no Monitor de Eficiência: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_to_history(self, metrics):
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: history = []
        
        # Manter apenas um registro por dia
        history = [h for h in history if h.get('date') != metrics['date']]
        history.append(metrics)
        
        # Manter últimos 12 registros
        history = history[-12:]
        
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def get_latest_metrics(self):
        if not os.path.exists(self.history_path):
            return None
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return history[-1] if history else None
        except: return None
