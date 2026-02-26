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
        Compara o ranking consolidado com a Eficiência Bruta (CVLI) + Eventos Exógenos.
        """
        if self.orchestrator is None:
            print("⚠️ [Monitor] Orquestrador não inicializado.")
            return None

        try:
            print("🧠 [Monitor] Iniciando avaliação de eficiência...")
            # 1. Obter Predições Consolidadas
            scores_map = self.orchestrator.get_combined_risk(None)
            print(f"🧠 [Monitor] Scores gerados para {len(scores_map)} localidades.")
            
            # 2. Construir Ground Truth Híbrido (Bruta + Exógena)
            ground_truth = {}
            total_brute = 0
            total_exo = 0

            # 2a. EFICIÊNCIA BRUTA: Coletar CVLIs Reais dos últimos 7 dias nos modelos
            for r_name, spec in self.orchestrator.specialists.items():
                data = spec['data']
                nf = data['node_features']
                # Canal 0 é CVLI. Pegamos a soma dos últimos 7 passos de tempo do dataset
                recent_cvli = nf[:, -7:, 0].sum(axis=1)
                for i, row in data['nodes_gdf'].iterrows():
                    count = int(recent_cvli[i])
                    if count > 0:
                        loc_norm = normalize_name(str(row['name']))
                        ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + count
                        total_brute += count

            # 2b. COMPLEMENTO EXÓGENO: Carregar eventos recentes de arquivos
            event_files = [os.path.join(self.root, "data", "exogenous_events.json")]
            for f in os.listdir(os.path.join(self.root, "data")):
                if f.startswith("exogenous_events_") and f.endswith(".json"):
                    event_files.append(os.path.join(self.root, "data", f))
            
            today = datetime.now().date()
            window_start = today - timedelta(days=7)

            for file_path in event_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            events = json.load(f)
                            if not isinstance(events, list): continue
                            for e in events:
                                dstr = e.get('date') or e.get('event_date') or e.get('ingested_at')
                                if not dstr: continue
                                try:
                                    e_date = datetime.strptime(dstr[:10], '%Y-%m-%d').date()
                                    if e_date >= window_start:
                                        loc_raw = e.get('bairro') or e.get('location') or e.get('municipio')
                                        if loc_raw:
                                            loc_norm = normalize_name(str(loc_raw))
                                            ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + 1
                                            total_exo += 1
                                except: continue
                    except: continue

            print(f"📊 [Monitor] Ground Truth Final: {len(ground_truth)} localidades com eventos ({total_brute} Bruta + {total_exo} Exógena)")

            # 3. Preparar Grupos de Avaliação
            regions = {'global': list(scores_map.keys())}
            for r_name, spec in self.orchestrator.specialists.items():
                regions[r_name] = [normalize_name(n) for n in spec['data']['nodes_gdf']['name']]
            
            results = {
                "date": str(today), 
                "total_events": total_brute + total_exo,
                "brute_cvli": total_brute,
                "exogenous": total_exo
            }
            
            # 4. Avaliar cada Região
            for r_name, node_list in regions.items():
                r_scores = {n: scores_map[n] for n in node_list if n in scores_map}
                r_ground_truth = {n: count for n, count in ground_truth.items() if n in node_list}
                
                if not r_scores:
                    results[r_name] = {"status": "no_scores", "p5": 0.0, "p10": 0.0, "p20": 0.0}
                    continue
                
                if not r_ground_truth:
                    results[r_name] = {"status": "no_events", "p5": 0.0, "p10": 0.0, "p20": 0.0}
                    continue
                    
                r_ranking = sorted(r_scores.items(), key=lambda x: x[1], reverse=True)
                
                region_metrics = {
                    "total_nodes": len(r_scores),
                    "active_locations": len(r_ground_truth),
                    "total_events": sum(r_ground_truth.values())
                }
                
                for k in [5, 10, 20]:
                    k_adj = min(k, len(r_ranking))
                    top_k = [name for name, score in r_ranking[:k_adj]]
                    hits = [name for name in top_k if name in r_ground_truth]
                    precision = len(hits) / k_adj if k_adj > 0 else 0
                    region_metrics[f"p{k}"] = round(precision, 4)
                    region_metrics[f"hits{k}"] = hits
                
                results[r_name] = region_metrics
                print(f"✅ [Monitor] Região {r_name.upper()}: P10={region_metrics.get('p10', 0)*100:.1f}%")

            # 5. Salvar Histórico e Retornar
            self.save_to_history(results)
            return results
        except Exception as e:
            print(f"⚠️ Erro no Monitor de Eficiência: {e}")
            return None

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
