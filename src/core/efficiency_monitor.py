import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.core.orchestrator import normalize_name


def classify_monitor_event(event):
    text = ' '.join([
        str(event.get('type') or ''),
        str(event.get('natureza') or ''),
        str(event.get('description') or ''),
        str(event.get('descricao') or ''),
        str(event.get('resumo') or ''),
        str(event.get('raw_text') or ''),
    ]).upper()

    conflict_keywords = (
        'HOMICID', 'CHACINA', 'EXECUC', 'LESAO A BALA', 'LESÃO A BALA',
        'DESLOCAMENTO FORCADO', 'DESLOCAMENTO FORÇADO', 'EXPULSAO DE MORADORES',
        'EXPULSÃO DE MORADORES', 'ACHADO DE CADAVER', 'ACHADO DE CADÁVER'
    )
    qualified_police_keywords = (
        'TRAFICO', 'TRÁFICO', 'ENTORPEC', 'DROGA', 'APREENS', 'ARMA', 'FUZIL',
        'PISTOLA', 'REVOLV', 'PORTE ILEGAL', 'MANDADO DE PRIS', 'FLAGRANTE',
        'DESARTIC', 'LIDER', 'LIDERANCA', 'CHEFE', 'COMANDO'
    )
    admin_keywords = (
        'VEICULO LOCALIZADO', 'VEÍCULO LOCALIZADO', 'RECUPERAD', 'LOCALIZAD',
        'CELULAR COM ALERTA', 'CELULAR COM QUEIXA', 'RECEPTA', 'CONDUZID',
        'DIRECAO PERIGOSA', 'DIREÇÃO PERIGOSA', 'QUEIXA DE ROUBO', 'CLONADO'
    )

    if any(keyword in text for keyword in conflict_keywords):
        return 'conflict'
    if any(keyword in text for keyword in qualified_police_keywords):
        return 'qualified_police'
    if any(keyword in text for keyword in admin_keywords) or event.get('is_suppression'):
        return 'administrative_police'
    return 'other'

class EfficiencyMonitor:
    """
    Monitor de Eficiência Report Preview: Avalia o poder preditivo global do modelo.
    Compara o ranking consolidado de todas as 299 localidades monitoradas com os eventos reais.
    """
    def __init__(self, project_root, orchestrator, nodes_gdf, model_mode="stgat"):
        self.root = project_root
        self.orchestrator = orchestrator
        self.nodes_gdf = nodes_gdf
        self.model_mode = model_mode
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
            if self.model_mode == "stgat_v5" and hasattr(self.orchestrator, "get_combined_risk_stgat_v5"):
                scores_map = self.orchestrator.get_combined_risk_stgat_v5(None)
            else:
                scores_map = self.orchestrator.get_combined_risk(None)
            print(f"🧠 [Monitor] Scores gerados para {len(scores_map)} localidades.")
            monitored_nodes = set(scores_map.keys())
            
            # 2. Construir Ground Truth Híbrido (Bruta + Exógena)
            ground_truth = {}
            total_brute = 0
            total_exo = 0
            assigned_brute = 0
            assigned_exo = 0
            unmapped_exo = 0
            unmapped_exo_samples = []

            # 2a. EFICIÊNCIA BRUTA: Coletar CVLIs Reais dos últimos N passos de tempo
            # Interior tem eventos muito esparsos: usar janela mais longa (28 passos)
            # Fortaleza/RMF usam 14 passos (sinal mais denso, janela menor é suficiente)
            WINDOW_BY_REGION = {'fortaleza': 14, 'rmf': 28, 'interior': 28}
            
            # --- AJUSTE DE DATA DE REFERÊNCIA (ANTI-CALMARIA) ---
            # Se a base de dados estiver defasada, usamos a última data da base como 'hoje' para a avaliação.
            last_base_date = datetime.now().date()
            if hasattr(self.orchestrator, 'dates') and self.orchestrator.dates is not None:
                try:
                    d_val = self.orchestrator.dates[-1]
                    if isinstance(d_val, str):
                        last_base_date = datetime.strptime(d_val[:10], '%Y-%m-%d').date()
                    else:
                        last_base_date = d_val
                    print(f"📊 [Monitor] Avaliando eficácia com base na data: {last_base_date}")
                except: pass

            today = last_base_date
            # ---------------------------------------------------

            for r_name, spec in self.orchestrator.specialists.items():
                data = spec['data']
                nf = data['node_features']
                window = WINDOW_BY_REGION.get(r_name, 14)
                # Canal 0 é CVLI. Pegamos a soma dos últimos N passos de tempo do dataset
                recent_cvli = nf[:, -window:, 0].sum(axis=1)
                for i, row in data['nodes_gdf'].iterrows():
                    count = int(recent_cvli[i])
                    if count > 0:
                        loc_norm = normalize_name(str(row['name']))
                        ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + count
                        total_brute += count
                        if loc_norm in monitored_nodes:
                            assigned_brute += count

            # 2b. COMPLEMENTO EXÓGENO: Carregar eventos recentes de arquivos
            event_files = [os.path.join(self.root, "data", "exogenous_events.json")]
            
            window_start = last_base_date - timedelta(days=7)
            total_exo_conflict = 0
            total_police_qualified = 0
            total_police_admin = 0
            assigned_police_qualified = 0
            unmapped_police_qualified = 0
            seen_events = set()

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
                                        dedupe_key = (
                                            dstr[:19],
                                            str(e.get('bairro') or e.get('municipio') or ''),
                                            str(e.get('natureza') or ''),
                                            str(e.get('descricao') or e.get('raw_text') or e.get('resumo') or '')
                                        )
                                        if dedupe_key in seen_events:
                                            continue
                                        seen_events.add(dedupe_key)

                                        loc_raw = e.get('bairro') or e.get('location') or e.get('municipio')
                                        event_class = classify_monitor_event(e)
                                        if event_class == 'administrative_police':
                                            total_police_admin += 1
                                            continue
                                        if loc_raw:
                                            loc_norm = normalize_name(str(loc_raw))
                                            if event_class == 'conflict':
                                                ground_truth[loc_norm] = ground_truth.get(loc_norm, 0) + 1
                                                total_exo += 1
                                                total_exo_conflict += 1
                                                if loc_norm in monitored_nodes:
                                                    assigned_exo += 1
                                                else:
                                                    unmapped_exo += 1
                                                    if len(unmapped_exo_samples) < 10:
                                                        unmapped_exo_samples.append(str(loc_raw))
                                            elif event_class == 'qualified_police':
                                                total_police_qualified += 1
                                                if loc_norm in monitored_nodes:
                                                    assigned_police_qualified += 1
                                                else:
                                                    unmapped_police_qualified += 1
                                except: continue
                    except: continue

            # 2c. TENSÃO LATENTE: Territórios com domínio de facção confirmado
            # NOTA: Não incluímos facções latentes no Ground Truth de Eficiência (P@K)
            # para não inflar artificialmente as métricas em regiões pequenas.
            # O monitor de Cobertura Territorial (Health Monitor) já audita isso separadamente.
            total_faction = 0
            
            # Print de diagnóstico para o log
            print(f"📊 [Monitor] Ground Truth Final: {len(ground_truth)} localidades com eventos ativos ({total_brute} CVLI + {total_exo} Exógena)")
            assigned_total = assigned_brute + assigned_exo
            unmapped_total = (total_brute + total_exo) - assigned_total
            print(
                f"🧾 [Monitor] Eventos mapeados ao ranking: {assigned_total}/{total_brute + total_exo} "
                f"(CVLI={assigned_brute}, Exógena={assigned_exo}, Não mapeados={unmapped_total})"
            )
            if unmapped_exo_samples:
                print(f"🧾 [Monitor] Amostra de exógenos não mapeados: {unmapped_exo_samples}")
            
            # Diagnóstico: quantos nomes do ranking NÃO matcharam no ground truth
            unmatched = [n for n in scores_map.keys() if n not in ground_truth]
            matched_pct = (1 - len(unmatched) / len(scores_map)) * 100 if scores_map else 0
            print(f"🔗 [Monitor] Match de nomes: {len(ground_truth)}/{len(scores_map)} ({matched_pct:.1f}% coverage)")

            # 3. Preparar Grupos de Avaliação
            regions = {'global': list(scores_map.keys())}
            for r_name, spec in self.orchestrator.specialists.items():
                regions[r_name] = [normalize_name(n) for n in spec['data']['nodes_gdf']['name']]
            
            results = {
                "date": str(today), 
                "total_events": total_brute + total_exo,
                "brute_cvli": total_brute,
                "exogenous": total_exo,
                "exogenous_conflict": total_exo_conflict,
                "police_qualified_actions_7d": total_police_qualified,
                "police_administrative_events_7d": total_police_admin,
                "assigned_total_events": assigned_total,
                "unmapped_total_events": unmapped_total,
                "assigned_brute_cvli": assigned_brute,
                "assigned_exogenous": assigned_exo,
                "unmapped_exogenous": unmapped_exo,
                "assigned_police_qualified_actions": assigned_police_qualified,
                "unmapped_police_qualified_actions": unmapped_police_qualified,
                "unmapped_exogenous_sample": unmapped_exo_samples,
            }
            
            # 4. Avaliar cada Região
            for r_name, node_list in regions.items():
                # Filtrar scores: apenas bairros que pertencem a esta região
                r_scores = {n: scores_map[n] for n in node_list if n in scores_map}
                
                # Filtrar ground truth: apenas eventos desta região
                r_ground_truth = {n: count for n, count in ground_truth.items() if n in node_list}
                
                if not r_scores:
                    results[r_name] = {"status": "no_scores", "p5": 0.0, "p10": 0.0, "p20": 0.0}
                    continue
                
                # Ranking regional isolado
                r_ranking = sorted(r_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Mínimo de eventos para avaliação (Reduzido para 1 para todas as regiões em períodos de baixa atividade)
                MIN_EVENTS = 1
                if len(r_ground_truth) < MIN_EVENTS:
                    results[r_name] = {
                        "status": f"insufficient_events ({len(r_ground_truth)}<{MIN_EVENTS})",
                        "p5": None, "p10": None, "p20": None,
                        "active_locations": len(r_ground_truth),
                        "total_nodes": len(r_scores),
                        "total_events": sum(r_ground_truth.values()),
                        "note": f"Menos de {MIN_EVENTS} localidades com eventos. Avaliação não confiável."
                    }
                    print(f"⚠️ [Monitor] Região {r_name.upper()}: SKIP — apenas {len(r_ground_truth)} localidades com eventos (mín={MIN_EVENTS}).")
                    if len(r_ground_truth) == 0:
                        print(f"ℹ️ [Monitor] DICA: Verifique se o dataset processado (.pkl) não está à frente dos dados reais do CSV.")
                    continue
                
                region_metrics = {
                    "total_nodes": len(r_scores),
                    "active_locations": len(r_ground_truth),
                    "total_events": sum(r_ground_truth.values()),
                    "ranking_top10": [name for name, _ in r_ranking[:10]],
                    "gt_sample": list(r_ground_truth.keys())[:15]
                }
                
                gt_count = len(r_ground_truth)
                for k in [5, 10, 20]:
                    k_adj = min(k, len(r_ranking))
                    top_k = [name for name, score in r_ranking[:k_adj]]
                    hits = [name for name in top_k if name in r_ground_truth]
                    
                    # Precisão@K: Fração do Top-K que contém eventos reais (Métrica do Treino)
                    precision_k = len(hits) / k_adj if k_adj > 0 else 0
                    
                    # Cobertura@K (Recall@K): Fração do Ground Truth capturada no Top-K
                    coverage_k = len(hits) / gt_count if gt_count > 0 else 0
                    
                    # Armazenamos a Precisão nos campos p5/p10/p20 para compatibilidade com o Dashboard
                    region_metrics[f"p{k}"] = round(precision_k, 4)
                    region_metrics[f"recall{k}"] = round(coverage_k, 4)
                    region_metrics[f"hits{k}"] = hits
                
                results[r_name] = region_metrics
                print(f"✅ [Monitor] Região {r_name.upper()}: P@10={region_metrics.get('p10', 0)*100:.1f}% | P@20={region_metrics.get('p20', 0)*100:.1f}% | GT={len(r_ground_truth)} zonas")

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
