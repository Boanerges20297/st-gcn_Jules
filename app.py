from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import os
import pickle
import json
import warnings
import logging
import unicodedata
from datetime import datetime, timedelta
import re

# --- Orquestrador Regional ST-GAT ---
try:
    from src.core.orchestrator import StateOrchestrator, normalize_name
    from src.core.efficiency_monitor import EfficiencyMonitor
    orchestrator = None 
except ImportError:
    # Fallback se o PYTHONPATH não incluir a raiz corretamente
    import sys
    sys.path.append(os.getcwd())
    from src.core.orchestrator import StateOrchestrator
    from src.core.efficiency_monitor import EfficiencyMonitor
    def normalize_name(text):
        if not isinstance(text, str): return ""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').upper().strip()
        import re
        return re.sub(r'\s*-\s*AIS.*$', '', text).strip()

warnings.filterwarnings('ignore')
# Configurando logs para garantir visibilidade no terminal
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache file for manager-harmonized explanations
CACHE_FILE = os.path.join(BASE_DIR, 'data', 'manager_explanations_cache.json')

import threading
import time

nodes_gdf = None
orchestrator = None
efficiency_monitor = None

def run_background_efficiency_monitor():
    """Tarefa em background que executa a cada 7 dias ou no start."""
    global efficiency_monitor
    # Aguarda o sistema inicializar completamente
    time.sleep(10)
    while True:
        if efficiency_monitor is not None:
            try:
                num_loc = len(nodes_gdf) if nodes_gdf is not None else 0
                print(f"\n" + "="*60)
                print(f"🛡️  [MONITOR DE EFICIÊNCIA] Iniciando Avaliação ({num_loc} localidades)")
                print("="*60)
                
                metrics = efficiency_monitor.run_evaluation()
                
                if metrics:
                    print(f"📅 Data da Avaliação: {metrics.get('date')}")
                    print(f"📊 Eventos Detectados: {metrics.get('total_events', 0)} ({metrics.get('brute_cvli', 0)} Brutos + {metrics.get('exogenous', 0)} Exógenos)")
                    
                    # Exibir Global
                    if 'global' in metrics:
                        m = metrics['global']
                        print(f"\n🌍 REGIONALIZAÇÃO: GLOBAL")
                        print(f"   P5:  {m.get('p5', 0)*100:.1f}% | Hits: {', '.join(m.get('hits5', []))}")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                        print(f"   P20: {m.get('p20', 0)*100:.1f}% | Hits: {', '.join(m.get('hits20', []))}")
                    
                    # Exibir Fortaleza
                    if 'fortaleza' in metrics:
                        m = metrics['fortaleza']
                        print(f"\n🏙️  REGIONALIZAÇÃO: FORTALEZA")
                        print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                    
                    # Exibir RMF e Interior se houver acertos
                    for reg in ['rmf', 'interior']:
                        if reg in metrics and metrics[reg].get('p10', 0) > 0:
                            m = metrics[reg]
                            reg_name = "REGIÃO METROPOLITANA" if reg == 'rmf' else "INTERIOR"
                            print(f"\n📍 REGIONALIZAÇÃO: {reg_name}")
                            print(f"   P10: {m.get('p10', 0)*100:.1f}% | Hits: {', '.join(m.get('hits10', []))}")
                    
                    print("\n" + "="*60 + "\n")
                else:
                    print("📊 [Monitor] Sem eventos suficientes para avaliação hoje.")
            except Exception as e:
                print(f"⚠️ [Monitor] Erro na thread de eficiência: {e}")
        
        # Dorme por 7 dias antes da próxima rodada (604800 segundos)
        time.sleep(604800)

def verify_date_consistency(event_date_str, last_base_date):
    """
    Verifica a consistência temporal.
    
    MODO PROTÓTIPO (Data Lag Tolerance):
    Aceita o evento se:
    1. For anterior ou igual à base do modelo (Consistência Histórica)
    2. OU Se for posterior à base mas anterior/igual a HOJE (Preenchimento do Gap de Atraso)
    
    Rejeita apenas se for > HOJE (Futuro Real).
    """
    if not event_date_str:
        return True # Sem data, aceita por segurança
        
    try:
        # Conversão robusta de strings
        if isinstance(event_date_str, str):
            e_date = datetime.strptime(event_date_str[:10], '%Y-%m-%d').date()
        elif hasattr(event_date_str, 'date'):
            e_date = event_date_str.date()
        else:
            e_date = event_date_str
            
        # Data de Hoje (Limite do Real)
        today = datetime.now().date()
        
        # Se o evento é futuro em relação ao tempo real, rejeita sempre
        if e_date > today:
            return False
            
        # Se não temos last_base_date, aceitamos pois é <= today
        if not last_base_date:
            return True
            
        # Lógica Original Estrita (Comentada para o Protótipo)
        # if isinstance(last_base_date, str):
        #     b_date = datetime.strptime(last_base_date[:10], '%Y-%m-%d').date()
        # elif hasattr(last_base_date, 'date'):
        #     b_date = last_base_date.date()
        # else:
        #     b_date = last_base_date
        # return e_date <= b_date
        
        # Lógica de Tolerância de Atraso (Prototype Mode)
        # Aceitamos o evento pois ele representa a realidade atual sobreposta ao modelo defasado
        return True 

    except Exception:
        return True # Em caso de erro, permitimos a inclusão

def archive_old_exogenous_events():
    """
    Cria um 'arquivo morto' dos eventos exógenos que ultrapassam os últimos 7 dias.
    O arquivo é salvo como 'data/exogenous_events_(data_limite).json'.

    Correção: alguns registros não tinham campo `date` mas tinham `ingested_at` e `timestamp` (hora).
    Neste caso, combinamos a data de `ingested_at` com o `timestamp` da ocorrência para obter a data
    do evento. Se não for possível extrair uma data, consideramos o evento recente e NÃO o arquivamos.
    """
    exogenous_file = os.path.join(BASE_DIR, "data", "exogenous_events.json")
    if not os.path.exists(exogenous_file):
        return

    try:
        with open(exogenous_file, 'r', encoding='utf-8') as f:
            events = json.load(f)

        if not events:
            return

        cutoff_date = (datetime.now() - timedelta(days=7)).date()

        old_events = []
        current_events = []

        for e in events:
            event_date = None

            # 1) Prefer explicit 'date' field if it's a full date string (YYYY-MM-DD)
            dval = e.get('date') or e.get('event_date')
            if isinstance(dval, str):
                try:
                    # if it's only a time like '22:10', skip here
                    if re.match(r'^\d{2}:\d{2}$', dval.strip()):
                        dval = None
                    else:
                        # accept ISO-like strings, take leading YYYY-MM-DD
                        event_date = datetime.strptime(dval.strip()[:10], '%Y-%m-%d').date()
                except Exception:
                    event_date = None

            # 2) If no full date, try combining 'ingested_at' (datetime) with 'timestamp' (HH:MM)
            if event_date is None:
                ing = e.get('ingested_at')
                ts = e.get('timestamp')  # e.g. '22:10'
                try:
                    ing_dt = None
                    if isinstance(ing, str) and ing:
                        try:
                            ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            try:
                                ing_dt = datetime.fromisoformat(ing.strip())
                            except Exception:
                                ing_dt = None

                    if ing_dt:
                        # If timestamp present and looks like HH:MM, we could combine,
                        # but for archiving we only need the date portion.
                        event_date = ing_dt.date()
                except Exception:
                    event_date = None

            # 3) If still no date, treat as recent (do not archive)
            if event_date is None:
                current_events.append(e)
                continue

            # Decide to archive or keep
            if event_date < cutoff_date:
                old_events.append(e)
            else:
                current_events.append(e)

        if old_events:
            # Write archive files into a dedicated directory to avoid cluttering
            archives_dir = os.path.join(BASE_DIR, 'data', 'archives')
            os.makedirs(archives_dir, exist_ok=True)

            archive_filename = f"exogenous_events_{cutoff_date.isoformat()}.json"
            archive_path = os.path.join(archives_dir, archive_filename)

            # Write atomically: write to a temp file then replace
            try:
                tmp_path = archive_path + '.tmp'
                with open(tmp_path, 'w', encoding='utf-8') as af:
                    json.dump(old_events, af, indent=2, ensure_ascii=False)
                os.replace(tmp_path, archive_path)
            except Exception as e:
                print(f"⚠️ Erro ao gravar arquivo morto: {e}")

            # Update canonical events file atomically as well
            try:
                tmp_main = exogenous_file + '.tmp'
                with open(tmp_main, 'w', encoding='utf-8') as f:
                    json.dump(current_events, f, indent=2, ensure_ascii=False)
                os.replace(tmp_main, exogenous_file)
            except Exception as e:
                print(f"⚠️ Erro ao atualizar arquivo principal de eventos: {e}")

            print(f"📦 Arquivo morto criado: data/archives/{archive_filename} ({len(old_events)} eventos)")
            print(f"✅ Arquivo principal atualizado ({len(current_events)} eventos ativos)")

    except Exception as e:
        print(f"⚠️ Erro ao arquivar eventos exógenos: {e}")

def generate_daily_ranking_report():
    """
    Gera um relatório Markdown diário com o Top 20 de cada região.
    Utilizado para acompanhamento manual de eficiência e auditoria.
    """
    if orchestrator is None or nodes_gdf is None:
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    base_log_dir = os.path.join(BASE_DIR, "logs", "rankings")
    os.makedirs(base_log_dir, exist_ok=True)

    # Calculamos o risco atual (sem shocks para servir de baseline estável ou com os atuais)
    scores_map = orchestrator.get_combined_risk()
    
    regions = {
        'fortaleza': 'FORTALEZA (CAPITAL)',
        'rmf': 'REGIÃO METROPOLITANA',
        'interior': 'INTERIOR DO ESTADO'
    }

    for reg_key, reg_name in regions.items():
        filename = f"ranking_{today_str}_{reg_key}.md"
        filepath = os.path.join(base_log_dir, filename)

        # Se já existir o relatório de hoje, não sobrescrevemos (mantém o snapshot inicial)
        if os.path.exists(filepath):
            continue

        try:
            # Filtrar e ordenar bairros da região
            reg_results = []
            for i, row in nodes_gdf.iterrows():
                # Lógica de identificação de região similar ao api/risk
                r = str(row.get('regiao', 'fortaleza')).lower()
                if r == 'capital': r = 'fortaleza'
                
                name = str(row['name'])
                name_norm = normalize_name(name)
                
                # Sincronização RMF Oficial
                rmf_oficial = ['AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 'PACATUBA', 'PARAIPABA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 'SAO LUIS DO CURU', 'TRAIRI']
                if name_norm in rmf_oficial: r = 'rmf'
                
                if r == reg_key:
                    score = float(scores_map.get(name_norm, 20.0))
                    reg_results.append({
                        'name': name,
                        'score': score,
                        'faction': str(row.get('faction', 'N/A'))
                    })

            # Ordenar por Score (Top 20)
            reg_results.sort(key=lambda x: x['score'], reverse=True)
            top_20 = reg_results[:20]

            # Escrever o arquivo Markdown
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 🛡️ Relatório de Risco Diário - {reg_name}\n")
                f.write(f"**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"**Estado da Base Histórica:** {orchestrator.dates[-1] if hasattr(orchestrator, 'dates') else 'N/A'}\n\n")
                f.write("| Pos | Localidade | Risco (%) | Facção Predominante |\n")
                f.write("|:---:|:---|:---:|:---:|\n")
                for idx, res in enumerate(top_20):
                    f.write(f"| {idx+1} | {res['name']} | {res['score']:.2f}% | {res['faction']} |\n")
                
                f.write(f"\n\n*Nota: Este ranking reflete o estado de inteligência do modelo no início do dia operacional.*")

            print(f"📄 Relatório gerado: {filename}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar relatório {reg_key}: {e}")

def load_data_and_models():
    global nodes_gdf, orchestrator, efficiency_monitor
    # Limpeza de eventos exógenos antigos
    archive_old_exogenous_events()
    
    path = os.path.join(BASE_DIR, "data", "processed", "processed_graph_data_global.pkl")
    if not os.path.exists(path):
        print(f"❌ Erro: Metadados não encontrados em {path}. Verifique se o arquivo existe e está no local correto.")
    
    if os.path.exists(path):
        with open(path, "rb") as f:
            nodes_gdf = pickle.load(f).get("nodes_gdf")
            print(f"✅ Metadados: {len(nodes_gdf)} localidades.")

    try:
        orchestrator = StateOrchestrator(BASE_DIR)
        print("✅ Motor de Inteligência ST-GAT Ativo.")
        
        # Iniciar Monitor de Eficiência e Relatórios
        efficiency_monitor = EfficiencyMonitor(BASE_DIR, orchestrator, nodes_gdf)
        generate_daily_ranking_report()
        
        # Disparar Monitor em Segundo Plano (Thread Paralela)
        threading.Thread(target=run_background_efficiency_monitor, daemon=True).start()
    except Exception as e:
        print(f"❌ Erro Motor: {e}")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/connections')
def connections(): return render_template('connections.html')

@app.route('/api/risk')
def get_risk():
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    try:
        # --- Build exogenous_shocks from recent exogenous events ---
        exogenous_shocks = {}
        try:
            # Carregar de ambos os arquivos potenciais para garantir cobertura
            # Use only the canonical exogenous events file. Previously the
            # geocoded variant was written as a duplicate which created
            # multiple files and confusion. Keep a single source of truth.
            exo_files = ['exogenous_events.json']
            all_raw_events = []
            for f_name in exo_files:
                f_path = os.path.join(BASE_DIR, 'data', f_name)
                if os.path.exists(f_path):
                    try:
                        with open(f_path, 'r', encoding='utf-8') as xf:
                            all_raw_events.extend(json.load(xf) or [])
                    except: pass

            # Considerar eventos nos últimos 7 dias
            cutoff = datetime.now().date() - timedelta(days=7)
            
            # Limite superior baseado nos dados do orquestrador (evita inconsistência teórica)
            last_date_base = None
            if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
                last_date_base = orchestrator.dates[-1]

            # Tipos que sempre são CRÍTICOS (Canal 25) - Incluindo sinais de violência extrema
            CRITICAL_TYPES = [
                'leader_transfer', 'faction_conflict', 'territory_dispute', 
                'confronto', 'execucao', 'chacina', 'tortura', 'homicidio_com_sinais_de_faccao'
            ]
            
            # Tipos que são de SUPRESSÃO (Canal 23) - Ação Policial Positiva
            SUPPRESSION_TYPES = [
                'apreensao', 'prisao', 'recuperacao_veiculo', 'cumprimento_mandado',
                'abordagem_positiva', 'desarticulacao_grupo'
            ]

            # Small helper: RMF cities (covers metro municipalities frequently used in reports)
            RMF_CITIES = {'MARACANAU', 'CAUCAIA', 'AQUIRAZ', 'PACATUBA', 'PINDORETAMA', 'ITAITINGA', 'GUAIUBA', 'CHOROZINHO', 'MARANGUAPE'}

            for ev in all_raw_events:
                try:
                    # --- Verificação de Consistência Teórica (Não ver o futuro) ---
                    ev_date_str = ev.get('date') or ev.get('event_date')
                    if not verify_date_consistency(ev_date_str, last_date_base):
                        continue # Pula evento futuro em relação ao estado do modelo

                    # Extração e Normalização da Localidade
                    bairro_raw = (ev.get('bairro') or '').strip()
                    municipio_raw = (ev.get('municipio') or '').strip()
                    # If a bairro is present, target that specific node; otherwise expand by municipio->region
                    targets = []  # list of normalized node names to apply this shock to
                    if bairro_raw:
                        targets = [normalize_name(str(bairro_raw))]
                    elif municipio_raw:
                        mun_up = municipio_raw.upper()
                        # Determine high-level region from municipality name
                        if 'FORTALEZA' in mun_up:
                            region_key = 'fortaleza'
                        elif mun_up in RMF_CITIES or any(c in mun_up for c in RMF_CITIES):
                            region_key = 'rmf'
                        else:
                            region_key = 'interior'

                        # Expand to all node names that belong to that region
                        try:
                            for i, row in nodes_gdf.iterrows():
                                if str(row.get('regiao', '')).lower() == region_key:
                                    targets.append(normalize_name(row['name']))
                        except Exception:
                            # Fallback: use municipality string as single target
                            targets = [normalize_name(municipio_raw)]
                    else:
                        continue

                    # Intensidade e Criticidade
                    ev_type = str(ev.get('type') or ev.get('natureza') or '').lower()
                    description = str(ev.get('description') or ev.get('resumo') or '').lower()
                    
                    # Classificação de Supressão e Ajuste de Intensidade Técnica
                    is_supp = (ev_type in SUPPRESSION_TYPES) or ('apreen' in ev_type) or ('pris' in ev_type)
                    
                    # Se for supressão, calibramos a intensidade pelo impacto
                    if is_supp:
                        if any(w in description for w in ['fuzil', 'metralhadora', 'fuzi', '7.62', '5.56']):
                            intensity = 1.0
                        elif any(w in description for w in ['lider', 'chefe', 'frente', 'comando']):
                            intensity = 0.9
                        elif any(w in description for w in ['pistola', 'revolver', 'arma de fogo']):
                            intensity = 0.7
                        elif any(w in description for w in ['quilos', 'kg', 'grande quantidade', 'deposito']):
                            intensity = 0.6
                        elif any(w in description for w in ['veiculo', 'carro', 'moto', 'recuperad']):
                            intensity = 0.4
                        else:
                            intensity = float(ev.get('intensity', 0.3))
                    else:
                        intensity = float(ev.get('intensity', 0.5))
                    
                    # Decisão de Canal: Canal 25 se tipo for crítico, intensidade > 0.7 
                    # OU se a descrição contiver palavras-chave de alerta máximo
                    is_critical = (ev_type in CRITICAL_TYPES) or (not is_supp and intensity > 0.7) or \
                                  ('execuc' in description) or ('facç' in description) or \
                                  ('morte' in description and 'facç' in description)
                    
                    # Apply/update shock for all resolved targets (single bairro or expanded region nodes)
                    for loc_norm in targets:
                        if loc_norm not in exogenous_shocks:
                            exogenous_shocks[loc_norm] = {
                                'intensity': intensity,
                                'is_critical': is_critical,
                                'is_suppression': is_supp
                            }
                        else:
                            if is_critical:
                                exogenous_shocks[loc_norm]['is_critical'] = True
                            if is_supp:
                                exogenous_shocks[loc_norm]['is_suppression'] = True
                            if intensity > exogenous_shocks[loc_norm]['intensity']:
                                exogenous_shocks[loc_norm]['intensity'] = intensity
                except: continue

            if not exogenous_shocks:
                exogenous_shocks = None
        except Exception as e:
            print(f"Erro ao processar shocks no app.py: {e}")
            exogenous_shocks = None

        scores_map, trends_map = orchestrator.get_combined_risk(exogenous_shocks, return_trends=True)
        results = []
        meta = {'counts': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}}
        all_scores = []
        
        # Prepare per-region accumulators
        region_buckets = {}
        # Contadores por região para o frontend
        region_stats = {
            'fortaleza': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0},
            'rmf': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0},
            'interior': {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}
        }

        # Carregar cache de explicações do gestor para uso no dashboard
        manager_cache = {}
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                    manager_cache = json.load(cf) or {}
        except: pass

        for i, row in nodes_gdf.iterrows():
            name = str(row['name'])
            name_norm = normalize_name(name)
            score = float(scores_map.get(name_norm, 20.0))
            trend = trends_map.get(name_norm, 'stable')
            
            if np.isnan(score) or np.isinf(score): score = 20.0
            
            # Identificação de Região e Status (mantido)
            reg = str(row.get('regiao', 'fortaleza')).lower()
            if reg == 'capital': reg = 'fortaleza'
            rmf_oficial = ['AQUIRAZ', 'CASCAVEL', 'CAUCAIA', 'CHOROZINHO', 'EUSEBIO', 'GUAIUBA', 'HORIZONTE', 'ITAITINGA', 'MARACANAU', 'MARANGUAPE', 'PACAJUS', 'PACATUBA', 'PARAIPABA', 'PARACURU', 'PINDORETAMA', 'SAO GONCALO DO AMARANTE', 'SAO LUIS DO CURU', 'TRAIRI']
            if name_norm in rmf_oficial: reg = 'rmf'

            if score >= 90: 
                status, css, color = 'CRÍTICO', 'risk-critico', '#8B0000'
                if reg in region_stats: region_stats[reg]['crítico'] += 1
                meta['counts']['crítico'] += 1
            elif score >= 80: 
                status, css, color = 'ALTO', 'risk-alto', '#E63946'
                if reg in region_stats: region_stats[reg]['alto'] += 1
                meta['counts']['alto'] += 1
            elif score >= 50: 
                status, css, color = 'MODERADO', 'risk-moderado', '#F4A261'
                if reg in region_stats: region_stats[reg]['moderado'] += 1
                meta['counts']['moderado'] += 1
            else: 
                status, css, color = 'BAIXO', 'risk-baixo', '#A8DADC'
                if reg in region_stats: region_stats[reg]['baixo'] += 1
                meta['counts']['baixo'] += 1
            
            if reg not in region_buckets: region_buckets[reg] = []
            
            # --- EXTRAÇÃO DE MÉTRICAS REAIS (DADOS BRUTOS DO MODELO) ---
            node_metrics = {
                'cvli_7d': 0,
                'tension': round(float(row.get('tension_index', 0)), 2),
                'events_count': 0,
                'event_types': [],
                'spatial_influence': score >= 80
            }
            
            # Crimes Reais
            current_spec = orchestrator.specialists.get(reg)
            if current_spec:
                try:
                    local_idx = next(idx for idx, r in current_spec['data']['nodes_gdf'].iterrows() if normalize_name(r['name']) == name_norm)
                    node_metrics['cvli_7d'] = int(current_spec['data']['node_features'][local_idx, -7:, 0].sum())
                except: pass

            # Eventos de Inteligência Reais
            if exogenous_shocks and name_norm in exogenous_shocks:
                # Filtrar eventos reais para listar os tipos
                exo_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
                if os.path.exists(exo_path):
                    try:
                        with open(exo_path, 'r', encoding='utf-8') as ef:
                            all_ev = json.load(ef)
                            node_evs = [e for e in all_ev if normalize_name(e.get('bairro','')) == name_norm or normalize_name(e.get('municipio','')) == name_norm]
                            node_metrics['events_count'] = len(node_evs)
                            node_metrics['event_types'] = list(set([e.get('natureza') or e.get('type') for e in node_evs]))
                    except: pass

            # Verificar cache de IA (apenas se existir)
            node_key = str(i)
            rich_explanation = manager_cache.get(node_key, {}).get('manager_text')

            all_scores.append(score)
            results.append({
                'node_id': i, 'name': name, 'clean_name': name_norm,
                'risk_score': score, 'status_label': status, 'css_class': css,
                'color': color, 'trend': trend, 
                'metrics': node_metrics,
                'reasons_rich': rich_explanation,
                'faction': str(row.get('faction', 'N/A')), 'region_type': reg
            })
            region_buckets[reg].append(results[-1])

        # Adicionar Ranking Info para o Frontend
        if all_scores:
            meta['stats_overall_mean'] = float(np.mean(all_scores))
            meta['ranking_info'] = {
                'top_1_percent_threshold': float(np.percentile(all_scores, 99)),
                'top_5_percent_threshold': float(np.percentile(all_scores, 95)),
                'top_10_percent_threshold': float(np.percentile(all_scores, 90))
            }
        else:
            meta['ranking_info'] = {'top_1_percent_threshold': 99, 'top_5_percent_threshold': 95, 'top_10_percent_threshold': 90}

        # --- Métricas focadas no gestor: confiança do ranking e "temperatura do estado" ---
        try:
            scores_arr = np.array(all_scores) if all_scores else np.array([20.0])
            s_mean = float(np.mean(scores_arr))
            s_std = float(np.std(scores_arr))
            s_min = float(np.min(scores_arr))
            s_max = float(np.max(scores_arr))

            # Ordenar scores para estatísticas de topo
            sorted_scores_arr = np.sort(scores_arr)[::-1]
            
            # 1. Pressão nos Hotspots (Top 5 Mean)
            top5_scores = sorted_scores_arr[:5]
            meta['stats_top5_mean'] = float(np.mean(top5_scores)) if len(top5_scores) > 0 else s_mean
            
            # 2. Alerta do Top 10 (Top 10 Mean)
            top10_scores = sorted_scores_arr[:10]
            meta['stats_top10_mean'] = float(np.mean(top10_scores)) if len(top10_scores) > 0 else s_mean
            
            # 3. Corte de Prioridade (Mínimo do Top 5)
            meta['stats_top5_min'] = float(np.min(top5_scores)) if len(top5_scores) > 0 else s_min
            
            # 4. Volatilidade Geral (STD)
            meta['stats_overall_std'] = s_std

            # Separação entre top10 e média geral — indica clareza do ranking
            top10_threshold = int(np.percentile(scores_arr, 90)) if len(scores_arr) > 1 else s_mean
            top10_mean = float(np.mean([v for v in scores_arr if v >= top10_threshold])) if len(scores_arr) > 0 else s_mean
            separation = top10_mean - s_mean

            # Confiança heurística recalibrada
            denom = (s_max - s_min) if (s_max - s_min) > 0 else 1.0
            std_norm = min(1.0, s_std / (s_mean + 1e-6)) # Relativo à média
            sep_norm = min(1.0, separation / (s_std + 1e-6)) # Quantos desvios o top 10 está acima

            # Novo cálculo: Base de 65%, bônus por separação, penalidade leve por volatilidade
            confidence_score = 0.65 + (0.30 * sep_norm) - (0.15 * std_norm)
            confidence_score = max(0.4, min(0.98, confidence_score))
            confidence_pct = round(confidence_score * 100.0, 1)

            if confidence_pct >= 80:
                confidence_label = 'Alta'
            elif confidence_pct >= 60:
                confidence_label = 'Moderada'
            elif confidence_pct >= 40:
                confidence_label = 'Baixa'
            else:
                confidence_label = 'Muito baixa'

            confidence_explanation = (
                f"Confiança estimada em {confidence_pct}% baseada em separação dos top {max(1,int(len(scores_arr)*0.1))}% "
                f"e estabilidade dos scores (desvio padrão {s_std:.2f})."
            )

            # Temperatura do estado (visão gerencial): mapeia média para níveis claros
            state_pct = round(s_mean, 1)
            if state_pct >= 90:
                temp_label = 'Crítico'
                temp_color = '#8B0000'
                recommendation = 'Intervenção imediata e mobilização de recursos.'
            elif state_pct >= 70:
                temp_label = 'Muito Quente'
                temp_color = '#E63946'
                recommendation = 'Aumentar vigilância e priorizar ações no top 10.'
            elif state_pct >= 50:
                temp_label = 'Quente'
                temp_color = '#F4A261'
                recommendation = 'Reforçar monitoramento e revisar alocação de recursos.'
            elif state_pct >= 30:
                temp_label = 'Morno'
                temp_color = '#A8DADC'
                recommendation = 'Manter operações regulares e monitorar tendências.'
            else:
                temp_label = 'Frio'
                temp_color = '#4CAF50'
                recommendation = 'Situação estável; operações normais.'

            meta['manager_view'] = {
                'confidence_pct': confidence_pct,
                'confidence_label': confidence_label,
                'confidence_explanation': confidence_explanation,
                'state_temperature_pct': state_pct,
                'state_temperature_label': temp_label,
                'state_temperature_color': temp_color,
                'recommendation': recommendation,
                'source': 'computed'
            }
        except Exception:
            meta['manager_view'] = {
                'confidence_pct': 50.0,
                'confidence_label': 'Moderada',
                'state_temperature_pct': meta.get('stats_overall_mean', 30.0),
                'state_temperature_label': 'Morno',
                'recommendation': 'Monitorar',
                'source': 'fallback'
            }

        # Build counts by region and top10 by region
        try:
            meta['counts_by_region'] = {}
            meta['top10_by_region'] = {}
            for region_key, items in region_buckets.items():
                c = {'crítico': 0, 'alto': 0, 'moderado': 0, 'baixo': 0}
                for it in items:
                    sc = it.get('risk_score', 0)
                    if sc >= 90:
                        c['crítico'] += 1
                    elif sc >= 80:
                        c['alto'] += 1
                    elif sc >= 50:
                        c['moderado'] += 1
                    else:
                        c['baixo'] += 1
                meta['counts_by_region'][region_key] = c

                sorted_region = sorted(items, key=lambda x: x.get('risk_score', 0), reverse=True)
                meta['top10_by_region'][region_key] = [{
                    'name': r.get('name'), 'node_id': r.get('node_id'), 'risk_score': r.get('risk_score'),
                    'status_label': r.get('status_label'), 'region_type': r.get('region_type')
                } for r in sorted_region[:10]]
        except Exception:
            meta['counts_by_region'] = {}
            meta['top10_by_region'] = {}

        # Build Top10 list server-side so frontend doesn't need to re-derive ranking.
        try:
            sorted_results = sorted(results, key=lambda x: x.get('risk_score', 0), reverse=True)
            meta['top10'] = []
            for r in sorted_results[:10]:
                meta['top10'].append({
                    'name': r.get('name'),
                    'node_id': r.get('node_id'),
                    'risk_score': r.get('risk_score'),
                    'status_label': r.get('status_label'),
                    'region_type': r.get('region_type')
                })
        except Exception:
            meta['top10'] = []

            # --- CORREÇÃO: Adicionar Datas da Janela de Inteligência (Projeção 7 dias) ---
        try:
            if orchestrator is not None and hasattr(orchestrator, 'dates') and orchestrator.dates is not None:
                last_db_date = orchestrator.dates[-1]
                if isinstance(last_db_date, str):
                    last_db_dt = datetime.strptime(last_db_date[:10], '%Y-%m-%d')
                else:
                    last_db_dt = last_db_date
                
                # Início e Fim da Projeção (7 dias à frente da base)
                start_pred = last_db_dt + timedelta(days=1)
                end_pred = last_db_dt + timedelta(days=7)
                
                meta['start_cvli'] = str(orchestrator.dates[0])
                meta['last_date_base'] = last_db_dt.strftime('%d/%m/%Y')
                meta['prediction_window'] = f"{start_pred.strftime('%d/%m')} a {end_pred.strftime('%d/%m')}"
                meta['intelligence_label'] = f"Janela de Inteligência: {meta['prediction_window']} (Atualizada com Eventos de Hoje)"
                meta['window_cvli'] = len(orchestrator.dates)
                meta['model_architecture'] = "Deep ST-GAT Elite (Regionalizado)"
                meta['model_window_cvli'] = 120 # Nova janela de 120 dias para todos
                
                # Incluir Eficiência Recente do Monitor
                if efficiency_monitor:
                    meta['efficiency_metrics'] = efficiency_monitor.get_latest_metrics()
            else:
                meta['intelligence_label'] = "Janela de Inteligência: Projeção 7 dias (Tempo Real)"
                meta['last_date_base'] = 'N/A'
                meta['model_architecture'] = "ST-GAT Elite v3"
                meta['model_window_cvli'] = 120
        except Exception as e:
            print(f"Erro ao calcular datas de inteligência: {e}")
            meta['intelligence_label'] = "Janela de Inteligência: Ativa"

        # --- CORREÇÃO: Respeitar Filtro de Região nas caixas de resumo ---
        target_region = request.args.get('region', 'global').lower()
        if target_region != 'global' and target_region in meta.get('counts_by_region', {}):
            meta['counts'] = meta['counts_by_region'][target_region]
            # Envia apenas os resultados daquela região
            results = region_buckets.get(target_region, [])

        return jsonify({'meta': meta, 'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/polygons')
def get_polygons():
    features = []
    ais_files = [('fortaleza', 'AIS - CAPITAL.geojson'), ('rmf', 'AIS - METROPOLITANA.geojson'), ('interior', 'AIS - INTERIOR.geojson')]
    for reg, fname in ais_files:
        path = os.path.join(BASE_DIR, 'data', 'static', fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for feat in data.get('features', []):
                        feat['properties']['region_type'] = reg
                        features.append(feat)
            except: pass
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/model-update-status')
def model_status(): return jsonify({"status": "idle"})

@app.route('/api/anomaly_status')
def anomaly_status():
    """Retorna o status de anomalias calculado em tempo real sobre o estado atual do modelo."""
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'monitoring_active': False, 'error': 'Inicializando...'}), 503
    
    try:
        # 1. Obter scores atuais para cálculo de tensão
        scores_map = orchestrator.get_combined_risk()
        scores = list(scores_map.values())
        
        # 2. Cálculo da Tensão Estadual (Escala 0-10)
        # Baseado na volatilidade (STD) e na média do Top 5%
        if scores:
            s_mean = np.mean(scores)
            s_std = np.std(scores)
            top_mean = np.percentile(scores, 95)
            # Tensão sobe se a média do topo for alta e houver muita variação
            tension = min(10.0, (top_mean / 20.0) + (s_std / 10.0))
        else:
            tension = 0.0
            
        # Determinar Label
        if tension >= 7.5: label = 'CRÍTICO'
        elif tension >= 5.0: label = 'ALERTA'
        else: label = 'ESTÁVEL'

        # 3. Listar Eventos Ativos Reais (Janela de 7 dias)
        active_events = []
        try:
            exo_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
            if os.path.exists(exo_path):
                with open(exo_path, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                
                cutoff = (datetime.now() - timedelta(days=7)).date()
                last_date_base = orchestrator.dates[-1] if (orchestrator and hasattr(orchestrator, 'dates')) else None
                for e in events:
                    # Tenta pegar a data
                    try:
                        dstr = e.get('date', '') or e.get('event_date', '')
                        if not verify_date_consistency(dstr, last_date_base):
                            continue # Pula evento futuro
                            
                        if dstr and datetime.strptime(dstr[:10], '%Y-%m-%d').date() >= cutoff:
                            active_events.append({
                                'description': e.get('description') or e.get('resumo', 'Evento Crítico'),
                                'severity': float(e.get('intensity', 0.5))
                            })
                    except: continue
        except: pass

        # 4. Confiança (Precisão de Captura)
        # Heurística: Baseada na separação estatística do sinal
        confidence = 0.5 # Base
        if scores and s_std > 0:
            separation = (np.max(scores) - np.mean(scores)) / s_std
            confidence = min(0.98, 0.6 + (separation / 10.0))

        return jsonify({
            'monitoring_active': True,
            'anomaly_level': float(tension),
            'anomaly_risk_level': label,
            'active_events': active_events[:5], # Top 5 eventos
            'model_confidence': float(confidence),
            'last_check': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'monitoring_active': True, 'error': str(e), 'anomaly_level': 0.0})


@app.route('/api/explain/<int:node_id>')
def explain_node(node_id):
    """Retorna uma explicação resumida dos motivos de criticidade para um nó (região/localidade).
    Implementação leve que responde mesmo sem o gerador de explicações completo disponível.
    """
    if nodes_gdf is None or orchestrator is None:
        return jsonify({'error': 'Inicializando...'}), 503
    try:
        if node_id not in list(nodes_gdf.index):
            return jsonify({'error': 'node not found'}), 404

        row = nodes_gdf.loc[node_id]
        name = str(row.get('name', 'unknown'))
        name_norm = normalize_name(name)

        # Obter scores e preparar contexto
        scores_map = orchestrator.get_combined_risk()
        score_pct = float(scores_map.get(name_norm, 20.0))
        # ExplanationGenerator trabalha com escala 0-10
        score_10 = score_pct / 10.0

        # Construir ranking e estatísticas locais
        all_scores = []
        node_score_pairs = []
        for i, r in nodes_gdf.iterrows():
            nname = normalize_name(str(r.get('name', '')))
            s = float(scores_map.get(nname, 20.0))
            all_scores.append(s)
            node_score_pairs.append((i, s))

        # Determinar posição por score (1 = maior)
        sorted_by_score = sorted(node_score_pairs, key=lambda x: x[1], reverse=True)
        ranks = {nid: idx + 1 for idx, (nid, _) in enumerate(sorted_by_score)}
        rank_pos = ranks.get(node_id, len(sorted_by_score))

        # Definir tier legível esperado pelo generator
        pct_rank = rank_pos / max(1, len(sorted_by_score))
        if rank_pos <= 5:
            tier = 'top_5'
        elif pct_rank <= 0.2:
            tier = 'long_tail_20'
        elif pct_rank <= 0.5:
            tier = 'long_tail_50'
        else:
            tier = 'tail'

        # Nearby: escolher até 3 peers com scores próximos ou na mesma região
        nearby = []
        try:
            # Preferir mesmos region_type quando disponível
            region_type = str(row.get('region_type', '')).lower()
            peers = [nid for nid, s in node_score_pairs if nid != node_id and str(nodes_gdf.loc[nid].get('region_type','')).lower() == region_type]
            if not peers:
                # fallback: peers by score proximity
                peers = [nid for nid, s in sorted_by_score if nid != node_id]
            nearby = peers[:3]
        except Exception:
            nearby = [nid for nid, _ in sorted_by_score if nid != node_id][:3]

        # Events: carregar eventos exógenos se disponíveis
        events = []
        events_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        last_date_base = orchestrator.dates[-1] if (orchestrator and hasattr(orchestrator, 'dates')) else None
        try:
            if os.path.exists(events_path):
                with open(events_path, 'r', encoding='utf-8') as ef:
                    evts = json.load(ef)
                    # filtrar por localidade aproximada pelo nome e consistência de data
                    for e in evts:
                        e_date_str = e.get('date') or e.get('event_date')
                        if not verify_date_consistency(e_date_str, last_date_base):
                            continue
                            
                        if name_norm in normalize_name(e.get('title', '')) or name_norm in normalize_name(e.get('location','')):
                            events.append(e)
        except Exception:
            events = []

        # Temporal pattern heurística
        overall_mean = float(np.mean(all_scores)) if all_scores else 30.0
        temporal_pattern = 'Increasing' if score_pct > overall_mean else 'Stable'

        # Confidence heurística
        confidence = min(0.95, 0.6 + (score_10 / 20.0))

        # Criar contexto esperado por ExplanationGenerator
        try:
            from src.explanation_generator import ExplanationGenerator
            gen = ExplanationGenerator()
            context = {
                'score': score_10,
                'temporal_pattern': temporal_pattern,
                'nearby_nodes': nearby,
                'events': events,
                'confidence': float(confidence),
                'tier': tier
            }

            explanation = gen.explain_node_ranking(int(node_id), int(rank_pos), context)
            # Ajustar para incluir score original em percent
            explanation['risk_score_pct'] = float(score_pct)

            # Normalizar estrutura para frontend: garantir 'factors' como [{name, contribution}]
            def _norm_factor_item(it):
                if not isinstance(it, dict):
                    return None
                name = it.get('name') or it.get('factor') or it.get('label') or it.get('factor_name') or it.get('factor_label')
                contrib = it.get('contribution') or it.get('value') or it.get('score') or it.get('weight') or it.get('percentage')
                # try to coerce strings like '45%' or '0.45' to numeric percent
                try:
                    if isinstance(contrib, str):
                        if contrib.strip().endswith('%'):
                            contrib = float(contrib.strip().rstrip('%'))
                        else:
                            contrib = float(contrib)
                    if isinstance(contrib, float) and contrib <= 1.0:
                        contrib = round(contrib * 100.0, 1)
                    if isinstance(contrib, (int, float)):
                        contrib = round(float(contrib), 1)
                except Exception:
                    contrib = None
                return {'name': name or '', 'contribution': contrib if contrib is not None else 0.0}

            normalized = {}
            normalized['node_id'] = node_id
            normalized['name'] = name
            normalized['risk_score_pct'] = float(score_pct)
            normalized['confidence'] = float(explanation.get('confidence', confidence))

            # summary
            summary = explanation.get('summary') or explanation.get('text') or f'Risco estimado {score_pct:.1f}%. Principais fatores: ver detalhes.'
            normalized['summary'] = summary

            # factors normalization
            raw_factors = explanation.get('factors') or explanation.get('factors_list') or []
            factors = []
            if isinstance(raw_factors, list) and raw_factors:
                for it in raw_factors:
                    nf = _norm_factor_item(it)
                    if nf:
                        factors.append(nf)
            else:
                # try to infer from common keys
                for key in ('temporal_pattern', 'spatial_correlation', 'recent_events', 'historical_baseline'):
                    if key in explanation:
                        val = explanation.get(key)
                        try:
                            contrib = float(val)
                            if contrib <= 1.0:
                                contrib = contrib * 100.0
                        except Exception:
                            contrib = None
                        if contrib is not None:
                            pretty = key.replace('_', ' ').title()
                            factors.append({'name': pretty, 'contribution': round(contrib, 1)})

            normalized['factors'] = factors

            # caveats/notes
            caveats = explanation.get('caveats') or explanation.get('notes') or explanation.get('warnings') or []
            normalized['caveats'] = caveats if isinstance(caveats, list) else [caveats]

            # Mark source as generator
            normalized['explanation_available'] = True
            normalized['source'] = 'generator'

            # --- Manager harmonized text: cache-aware LLM call ---
            def _ensure_cache_dir():
                d = os.path.dirname(CACHE_FILE)
                if not os.path.exists(d):
                    try:
                        os.makedirs(d, exist_ok=True)
                    except Exception:
                        pass

            def _load_cache():
                try:
                    if os.path.exists(CACHE_FILE):
                        with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                            return json.load(cf) or {}
                except Exception:
                    pass
                return {}

            def _save_cache(c):
                try:
                    _ensure_cache_dir()
                    with open(CACHE_FILE, 'w', encoding='utf-8') as cf:
                        json.dump(c, cf, ensure_ascii=False, indent=2)
                except Exception as e:
                    logging.exception('Failed saving manager_text cache: %s', e)

            def _parse_event_date(ev):
                # Try common fields and return YYYY-MM-DD or None
                if not isinstance(ev, dict):
                    return None
                for key in ('date', 'event_date', 'ingested_at'):
                    v = ev.get(key)
                    if not v:
                        continue
                    try:
                        s = str(v).strip()
                        if len(s) >= 10 and re.match(r'\d{4}-\d{2}-\d{2}', s):
                            return s[:10]
                    except Exception:
                        continue
                return None

            try:
                # compute newest event date in context (YYYY-MM-DD) if available
                events_list = context.get('events') if isinstance(context.get('events'), list) else []
                max_event_date = None
                for ev in events_list:
                    try:
                        d = _parse_event_date(ev)
                        if d:
                            if (not max_event_date) or d > max_event_date:
                                max_event_date = d
                    except Exception:
                        continue

                cache = _load_cache()
                node_key = str(node_id)
                need_call = False
                if node_key in cache:
                    cached = cache[node_key]
                    cached_last = cached.get('last_event_date')
                    if max_event_date and (not cached_last or max_event_date > cached_last):
                        need_call = True
                else:
                    need_call = True

                if need_call:
                    # Build prompt for manager harmonization
                    try:
                        prompt = (
                            "Você é um assistente que reescreve explicações técnicas para um gestor municipal.\n"
                            "Recebe a explicação estruturada em JSON abaixo. Produza um parágrafo curto (2-4 frases) em português claro, "
                            "destacando os fatores principais, o nível de confiança e recomendação de ação. Seja objetivo e inclua o nome da localidade.\n\n"
                            "EXPLICAÇÃO JSON:\n" + json.dumps(normalized, ensure_ascii=False)
                        )

                        # Attempt to call the LLM service
                        try:
                            import src.llm_service as llmsvc
                            keys = llmsvc.get_gemini_api_keys()

                            harmonized = None
                            if keys:
                                try:
                                    harmonized = llmsvc._call_model_with_rotation(prompt, keys)
                                except Exception:
                                    harmonized = None

                            # If harmonized text produced, cache it
                            if harmonized:
                                cache[node_key] = {
                                    'manager_text': harmonized,
                                    'cached_at': datetime.utcnow().isoformat(),
                                    'last_event_date': max_event_date
                                }
                                _save_cache(cache)
                                normalized['manager_text'] = harmonized
                                normalized['manager_text_source'] = 'llm'
                            else:
                                # if no model output but cache has previous value, reuse it
                                if node_key in cache and cache[node_key].get('manager_text'):
                                    normalized['manager_text'] = cache[node_key].get('manager_text')
                                    normalized['manager_text_source'] = 'cache'
                        except Exception:
                            # fallback to cached if present
                            if node_key in cache and cache[node_key].get('manager_text'):
                                normalized['manager_text'] = cache[node_key].get('manager_text')
                                normalized['manager_text_source'] = 'cache'
                    except Exception:
                        logging.exception('Error preparing manager_text')
                else:
                    # reuse cached manager_text
                    if node_key in cache and cache[node_key].get('manager_text'):
                        normalized['manager_text'] = cache[node_key].get('manager_text')
                        normalized['manager_text_source'] = 'cache'
            except Exception:
                logging.exception('Manager text caching flow failed')

            return jsonify(normalized)
        except Exception as e:
            # Fail-safe: do not return HTTP error. Provide a consistent JSON response
            # indicating the detailed explanation is unavailable while preserving
            # the node id, name and score so the frontend can continue rendering.
            safe_resp = {
                'node_id': node_id,
                'name': name,
                'risk_score_pct': float(score_pct),
                'confidence': float(confidence),
                'summary': 'Explicação detalhada indisponível',
                'factors': [],
                'caveats': [],
                'explanation_available': False,
                'source': 'unavailable',
            }
            # Log the underlying exception for diagnostics without crashing
            logging.exception('ExplanationGenerator failed: %s', e)
            return jsonify(safe_resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/exogenous/parse', methods=['POST'])
def parse_exogenous():
    """Parse raw CIOPS-like text into structured events using llm_service.process_exogenous_text
    Expects JSON: { text: '...raw lines...' }
    Returns: { points: [ {bairro, municipio, resumo, ...}, ... ] }
    """
    try:
        payload = request.get_json(force=True) or {}
        text = payload.get('text') or payload.get('raw') or ''
        if not text or not text.strip():
            return jsonify({'error': 'empty_text'}), 400

        try:
            from src.llm_service import process_exogenous_text
        except Exception as e:
            # Friendly JSON response that frontend can render in a user-facing modal
            friendly = {
                'error': 'llm_service_unavailable',
                'title': 'Serviço de extração temporariamente indisponível',
                'message': 'Não foi possível processar automaticamente o texto de origem. Tente novamente em alguns minutos ou entre com os pontos manualmente.',
                'detail': str(e)
            }
            return jsonify(friendly), 503

        try:
            parsed = process_exogenous_text(text)
        except Exception as e:
            friendly = {
                'error': 'llm_processing_failed',
                'title': 'Erro ao processar o texto',
                'message': 'O servidor encontrou um problema ao tentar interpretar os dados. Tente novamente ou informe o problema ao suporte.',
                'detail': str(e)
            }
            return jsonify(friendly), 503

        # Return parsed items as 'points' for frontend compatibility
        return jsonify({'points': parsed, 'count': len(parsed)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/exogenous/save', methods=['POST'])
def save_exogenous():
    """Save parsed exogenous events to disk for downstream processing.
    Expects JSON: { points: [...], original_text: '...' }
    Writes to `data/exogenous_events.json` and `data/exogenous_events_geocoded.json`.
    """
    try:
        payload = request.get_json(force=True) or {}
        points = payload.get('points') or []
        original = payload.get('original_text', '')

        if not isinstance(points, list) or len(points) == 0:
            return jsonify({'error': 'no_points'}), 400

        # Normalize minimal fields and add ingest metadata
        for p in points:
            if isinstance(p, dict):
                p.setdefault('ingested_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                # Ensure `date` exists and is a YYYY-MM-DD string (EventManager expects a date)
                # Prefer an explicit full-date if provided; otherwise derive from `ingested_at`.
                # We'll try to produce a full datetime string in `date`.
                # Priority:
                # 1) If `date` already contains a full datetime/ISO -> keep as-is
                # 2) If `date` is only a YYYY-MM-DD and `timestamp` available -> combine
                # 3) If `date` is only a time (HH:MM[:SS]) -> combine with `ingested_at` date
                # 4) Else try to derive date from `ingested_at` and combine with `timestamp` if present

                def _normalize_time_part(t):
                    if not t or not isinstance(t, str):
                        return None
                    t = t.strip()
                    # HH:MM or H:MM or HH:MM:SS
                    m = re.match(r'^(\d{1,2}:\d{2})(:?\d{0,2})$', t)
                    if m:
                        hhmm = m.group(1)
                        rest = m.group(2) or ''
                        if rest and rest.startswith(':') and len(rest) == 3:
                            return hhmm + rest
                        else:
                            return hhmm + ':00'
                    # If already HH:MM:SS
                    m2 = re.match(r'^(\d{2}:\d{2}:\d{2})$', t)
                    if m2:
                        return m2.group(1)
                    return None

                date_val = p.get('date')
                ts_val = p.get('timestamp') or p.get('time') or p.get('hora')
                raw_text = (p.get('raw_text') or p.get('descricao') or '')

                # Try to extract an explicit time or date from the raw text when available.
                # Many source messages append the occurrence time (e.g. "- 22:10") at the end.
                try:
                    if raw_text and isinstance(raw_text, str):
                        # prefer the last HH:MM(:SS) pattern found in the raw text
                        mtime = re.findall(r"(\d{1,2}:\d{2}(?::\d{2})?)", raw_text)
                        if mtime and (not ts_val or len(str(ts_val).strip()) == 0):
                            # take last found time
                            ts_val = mtime[-1]

                        # Also try to find an explicit date in raw_text (YYYY-MM-DD or DD/MM/YYYY)
                        mdate = re.search(r"(\d{4}-\d{2}-\d{2})", raw_text)
                        if not mdate:
                            mdate = re.search(r"(\d{2}/\d{2}/\d{4})", raw_text)
                        if mdate:
                            extracted_date = mdate.group(1)
                        else:
                            extracted_date = None
                    else:
                        extracted_date = None
                except Exception:
                    extracted_date = None

                # If we have an extracted_date in DD/MM/YYYY format, normalize to YYYY-MM-DD
                if extracted_date and re.match(r'^\d{2}/\d{2}/\d{4}$', extracted_date):
                    try:
                        dparts = extracted_date.split('/')
                        extracted_date = f"{dparts[2]}-{dparts[1]}-{dparts[0]}"
                    except Exception:
                        pass

                final_dt = None

                # 1) If date_val already contains a date part (YYYY-MM-DD)
                if isinstance(date_val, str) and re.search(r'\d{4}-\d{2}-\d{2}', date_val):
                    # If it includes time, keep full; otherwise try to append timestamp
                    try:
                        # Try parsing full datetime first
                        try:
                            parsed = datetime.fromisoformat(date_val.strip())
                            final_dt = parsed.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            # If only date present like YYYY-MM-DD
                            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_val.strip()):
                                date_part = date_val.strip()
                                tnorm = _normalize_time_part(ts_val)
                                if tnorm:
                                    final_dt = f"{date_part} {tnorm}"
                                else:
                                    final_dt = f"{date_part} 00:00:00"
                            else:
                                # fallback: take leading YYYY-MM-DD and default time
                                date_part = date_val.strip()[:10]
                                final_dt = f"{date_part} 00:00:00"
                    except Exception:
                        final_dt = None

                # 2) If date_val looks like a time only (HH:MM) -> combine with ingested_at date
                if final_dt is None and isinstance(date_val, str) and re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', date_val.strip()):
                    tnorm = _normalize_time_part(date_val.strip())
                    ing = p.get('ingested_at')
                    ing_dt = None
                    if isinstance(ing, str) and ing:
                        try:
                            ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            try:
                                ing_dt = datetime.fromisoformat(ing.strip())
                            except Exception:
                                ing_dt = None
                    if ing_dt:
                        final_dt = f"{ing_dt.date().isoformat()} {tnorm}"

                # 3) If still none, but timestamp present -> prefer timestamp's full datetime if it includes a date,
                # otherwise combine the timestamp (time-only) with a sensible date.
                if final_dt is None and ts_val:
                    # If ts_val contains a full date (YYYY-MM-DD) or ISO, try parsing directly
                    try:
                        if isinstance(ts_val, str) and re.search(r'\d{4}-\d{2}-\d{2}', ts_val):
                            # normalize full datetime
                            try:
                                parsed = datetime.fromisoformat(ts_val.strip())
                                final_dt = parsed.strftime('%Y-%m-%d %H:%M:%S')
                            except Exception:
                                # try parse common formats
                                try:
                                    parsed = datetime.strptime(ts_val.strip(), '%Y-%m-%d %H:%M:%S')
                                    final_dt = parsed.strftime('%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    final_dt = None
                        else:
                            # time-only: normalize and combine with best-available date
                            tnorm = _normalize_time_part(ts_val)
                            if tnorm:
                                # Prefer an explicit date extracted from raw_text
                                if extracted_date:
                                    final_dt = f"{extracted_date} {tnorm}"
                                else:
                                    # else prefer provided date_val if it contains a date
                                    if isinstance(date_val, str) and re.search(r'\d{4}-\d{2}-\d{2}', str(date_val)):
                                        date_part = str(date_val).strip()[:10]
                                        final_dt = f"{date_part} {tnorm}"
                                    else:
                                        # fall back to ingested_at date
                                        ing = p.get('ingested_at')
                                        ing_dt = None
                                        if isinstance(ing, str) and ing:
                                            try:
                                                ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                                            except Exception:
                                                try:
                                                    ing_dt = datetime.fromisoformat(ing.strip())
                                                except Exception:
                                                    ing_dt = None
                                        if ing_dt:
                                            final_dt = f"{ing_dt.date().isoformat()} {tnorm}"
                    except Exception:
                        final_dt = None

                # 4) Fallback: derive date from ingested_at and set midnight time
                if final_dt is None:
                    ing = p.get('ingested_at')
                    if isinstance(ing, str) and ing:
                        try:
                            ing_dt = datetime.strptime(ing.strip(), '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            try:
                                ing_dt = datetime.fromisoformat(ing.strip())
                            except Exception:
                                ing_dt = None
                        if ing_dt:
                            final_dt = f"{ing_dt.date().isoformat()} 00:00:00"
                if final_dt:
                    p['date'] = final_dt
                else:
                    p['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if 'raw_text' not in p and original:
                    p['raw_text'] = original

        # Save raw parsed to exogenous_events.json
        raw_path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        try:
            if os.path.exists(raw_path):
                with open(raw_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f) or []
                if isinstance(existing, dict) and 'events' in existing:
                    existing_list = existing['events']
                elif isinstance(existing, list):
                    existing_list = existing
                else:
                    existing_list = []
            else:
                existing_list = []
        except Exception:
            existing_list = []

        # Append and write back
        existing_list.extend(points)
        try:
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(existing_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return jsonify({'error': 'write_failed', 'detail': str(e)}), 500

        # Note: do not create a separate 'geocoded' file here. The system
        # should maintain a single canonical `exogenous_events.json` file.
        # Other components that require geocoded/enriched data should read
        # this file and perform their own enrichment rather than relying
        # on a duplicate file being written here.

        return jsonify({'saved': len(points)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/manager_explanations/cache', methods=['GET'])
def get_manager_cache():
    """Return the full manager_explanations cache JSON (for debugging/inspection)."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                data = json.load(cf) or {}
        else:
            data = {}
        return jsonify({'cache': data})
    except Exception as e:
        logging.exception('Failed reading manager_explanations cache: %s', e)
        return jsonify({'error': 'cache_read_failed', 'detail': str(e)}), 500


@app.route('/api/manager_explanations/cache/<node_id>', methods=['DELETE'])
def delete_manager_cache_node(node_id):
    """Invalidate cached manager text for a specific node_id."""
    try:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as cf:
                    cache = json.load(cf) or {}
            except Exception:
                cache = {}
        else:
            cache = {}

        if str(node_id) in cache:
            del cache[str(node_id)]
            try:
                with open(CACHE_FILE, 'w', encoding='utf-8') as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.exception('Failed saving cache after delete: %s', e)
                return jsonify({'error': 'cache_write_failed', 'detail': str(e)}), 500
            return jsonify({'deleted': node_id})
        return jsonify({'deleted': None, 'reason': 'not_found'})
    except Exception as e:
        logging.exception('Error deleting cache node: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/manager_explanations/cache/clear', methods=['POST'])
def clear_manager_cache():
    """Clear the entire manager_explanations cache."""
    try:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as cf:
                json.dump({}, cf, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.exception('Failed clearing cache: %s', e)
            return jsonify({'error': 'cache_clear_failed', 'detail': str(e)}), 500
        return jsonify({'cleared': True})
    except Exception as e:
        logging.exception('Error clearing cache: %s', e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/exogenous-events')
def get_exogenous_events_list():
    """Retorna a lista de eventos exógenos para o dashboard estratégico."""
    try:
        path = os.path.join(BASE_DIR, 'data', 'exogenous_events.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f) or []
            # Inverter para mostrar os mais recentes primeiro
            return jsonify(list(reversed(data)))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/efficiency-latest')
def get_efficiency_latest():
    """Retorna as métricas mais recentes do monitor de eficiência."""
    try:
        if efficiency_monitor:
            latest = efficiency_monitor.get_latest_metrics()
            return jsonify(latest if latest else {})
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    load_data_and_models()
    print("\n" + "="*50)
    print("DASHBOARD CPRAIO PRONTO")
    print("ACESSE: http://localhost:5050")
    print("="*50 + "\n")
    # Usando 0.0.0.0 para maior compatibilidade, mas o link impresso é localhost
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=True)
