"""
====================================================================
SENTINELA — CHAMPION/CHALLENGER DINÂMICO (Fase 6 → Integração)
====================================================================
Intercala os scores do ST-GAT (champion) com o LGBM Lean V3
(challenger) para a região Fortaleza, usando P@10 recente como árbitro.

Comportamento:
  - Avalia ambos os modelos contra CVLI real dos últimos EVAL_DAYS
  - Blend: w_champion × ST-GAT + w_challenger × LGBM
  - Pesos ajustados automaticamente (suavização exponencial)
  - Se challenger não tiver signal → 100% champion (fallback seguro)
  - Loga cada rodada em logs/cc_decisions.jsonl

Integração no app.py (1 linha por site de chamada):
    from src.core.champion_challenger import ChampionChallenger
    cc = ChampionChallenger(BASE_DIR)           # no startup
    scores_map = cc.apply(scores_map)            # após get_combined_risk()

====================================================================
"""

import os, json, pickle, warnings, unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────
EVAL_DAYS      = 14   # janela de avaliação (dias) — 1 horizonte
MIN_CVLI_CC    = 2    # mínimo de CVLI reais para arbitrar
MIN_ADVANTAGE  = 0.03 # vantagem mínima do challenger para ativar (3pp)
ALPHA_EMA      = 0.3  # suavização exponencial dos pesos (low = mais lento p/ mudar)
MAX_CC_WEIGHT  = 0.50 # peso máximo que o challenger pode atingir
INIT_CC_WEIGHT = 0.0  # peso inicial do challenger (começa em 0% = 100% champion)

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO': 15.0, 'PORTE ILEGAL ART 14': 12.0,
    'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
    'APREENSAO DE ENTORPECENTES': 6.0, 'MANDADO DE PRISAO': 4.0,
    'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
    'VEICULO ROUBADO RECUPERADO': 2.5, 'VEICULO ROUBADO LOCALIZADO': 2.0,
    'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
}

def _norm(text):
    if not isinstance(text, str): return ""
    t = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def _normalize_scores(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _pk(scores, targets, k):
    """P@k entre dois arrays de scores."""
    top_pred = set(np.argsort(scores)[::-1][:k])
    top_real = set(np.argsort(targets)[::-1][:k])
    return len(top_pred & top_real) / k


# ─────────────────────────────────────────────────────────────────
class ChampionChallenger:
    """
    Blend dinâmico entre ST-GAT e LGBM Lean para a região Fortaleza.

    Uso no app.py:
        # No startup (junto com o orchestrator):
        from src.core.champion_challenger import ChampionChallenger
        cc = ChampionChallenger(BASE_DIR)

        # Após cada get_combined_risk():
        scores_map = cc.apply(scores_map)
    """

    def __init__(self, base_dir: str):
        self.base_dir    = base_dir
        self.model_path  = os.path.join(base_dir, "models", "active", "sentinela_v4_model.pkl")
        self.cc_log      = os.path.join(base_dir, "logs", "cc_decisions.jsonl")
        self.state_path  = os.path.join(base_dir, "data", "cc_state.json")
        self.csv_enrich  = os.path.join(base_dir, "data", "raw", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
        self.csv_tropa   = os.path.join(base_dir, "data", "raw", "ocorrencias_tropa_limpo_fortaleza.csv")
        self.latlon_file = os.path.join(base_dir, "data", "raw", "bairros_centros_latlong.json")

        self._ranker       = None
        self._feat_names   = None
        self._ewma_weights = None
        self._top_bairros  = None
        self._feats        = None
        self._dates        = None
        self._cvli_raw     = None

        # Pesos persistidos entre reinícios
        self._cc_weight   = INIT_CC_WEIGHT
        self._last_eval   = None
        self._load_state()
        self._load_challenger()

    # ── API pública ───────────────────────────────────────────────

    def apply(self, scores_map: dict) -> dict:
        """
        Recebe o dict {bairro_norm: score} do ST-GAT e retorna
        um novo dict com os scores de Fortaleza eventualmente
        blendados com o LGBM Lean.

        Bairros de outras regiões (RMF, interior) são retornados intactos.
        """
        if self._ranker is None or self._top_bairros is None:
            return scores_map  # challenger não disponível — fallback total

        try:
            self._ensure_features()
        except Exception as e:
            print(f"⚠️ [CC] Erro ao construir features: {e}")
            return scores_map

        # ── Avaliar periodicamente (no máximo 1x/hora) ──
        now = datetime.now()
        if self._last_eval is None or (now - self._last_eval).seconds > 3600:
            self._evaluate_and_update(scores_map)
            self._last_eval = now

        if self._cc_weight < 0.01:
            return scores_map  # champion domina completamente

        # ── Gerar scores do challenger ──
        challenger_scores = self._get_challenger_scores()
        if challenger_scores is None:
            return scores_map

        # ── Blend apenas nos bairros de Fortaleza ──
        result = dict(scores_map)
        champ_arr = np.array([scores_map.get(b, 0.0) for b in self._top_bairros])
        chal_arr  = np.array([challenger_scores.get(b, 0.0) for b in self._top_bairros])

        # Normalizar ambos para [0,1] antes do blend
        champ_norm = _normalize_scores(champ_arr) * 100.0
        chal_norm  = _normalize_scores(chal_arr)  * 100.0

        blended = ((1 - self._cc_weight) * champ_norm
                 + self._cc_weight        * chal_norm)

        for i, b in enumerate(self._top_bairros):
            result[b] = float(blended[i])

        return result

    def status(self) -> dict:
        """Retorna estado atual do CC para logging/dashboard."""
        return {
            "cc_weight":    round(self._cc_weight, 3),
            "champion_pct": round((1 - self._cc_weight) * 100, 1),
            "challenger_pct": round(self._cc_weight * 100, 1),
            "last_eval":    str(self._last_eval) if self._last_eval else "nunca",
            "challenger_available": self._ranker is not None,
        }

    # ── Internos ─────────────────────────────────────────────────

    def _load_challenger(self):
        """Carrega o modelo LGBM Lean do arquivo .pkl."""
        if not os.path.exists(self.model_path):
            print(f"⚠️ [CC] Modelo challenger não encontrado: {self.model_path}")
            return
        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            self._ranker       = payload["ranker"]
            self._feat_names   = payload["feat_names_lgbm"]
            self._ewma_weights = payload["ewma_weights"]
            self._top_bairros  = [_norm(b) for b in payload["top_bairros"]]
            print(f"✅ [CC] Challenger Sentinela V4 carregado ({len(self._top_bairros)} bairros Fortaleza)")
        except Exception as e:
            print(f"❌ [CC] Erro ao carregar challenger: {e}")

    def _load_state(self):
        """Restaura pesos persistidos de execuções anteriores."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    s = json.load(f)
                self._cc_weight = float(s.get("cc_weight", INIT_CC_WEIGHT))
                last = s.get("last_eval")
                self._last_eval = datetime.fromisoformat(last) if last else None
                print(f"✅ [CC] Estado restaurado: challenger={self._cc_weight*100:.0f}%")
            except Exception:
                pass

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "cc_weight": self._cc_weight,
                "last_eval": str(self._last_eval) if self._last_eval else None,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)

    def _ensure_features(self):
        """Constrói features se ainda não foram construídas."""
        if self._feats is not None:
            return
        self._feats, self._dates, self._cvli_raw = self._build_features()

    def _build_features(self):
        """Constrói a matriz de features (N×T) a partir dos CSVs."""
        START = pd.Timestamp("2024-02-01")
        top_bairros_raw = [b for b in self._top_bairros]  # já normalizados

        df = pd.read_csv(self.csv_enrich, low_memory=False)
        df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
        df["bairro"] = df["bairro"].apply(_norm)
        df["data"]   = pd.to_datetime(df["data"], errors="coerce")
        df           = df.dropna(subset=["data","bairro"])
        df["is_cvli"]= (df["tipo"] == "cvli").astype("int8")
        df["is_cvp"] = (df["tipo"] == "cvp").astype("int8")

        df_t = pd.read_csv(self.csv_tropa, low_memory=False, encoding="utf-8-sig")
        df_t["bairro"]     = df_t["bairro"].apply(_norm)
        df_t["data"]       = pd.to_datetime(df_t["data"], errors="coerce")
        df_t               = df_t.dropna(subset=["data","bairro"])
        df_t["peso_nat"]   = df_t["natureza"].str.upper().str.strip().map(lambda x: PESO_NATUREZA.get(x, 1.0))
        df_t["score_intel"]= (
            df_t["qtd_armas"]*15 + np.log1p(df_t["qtd_drogas"].fillna(0))*4
            + df_t["qtd_drogas_itens"]*2 + df_t["qtd_veiculos_apreendidos"]*3
            + df_t["peso_nat"]
        ).astype("float32")

        dates = pd.date_range(START, df["data"].max(), freq="D")
        N, T  = len(top_bairros_raw), len(dates)
        nm    = {b: i for i,b in enumerate(top_bairros_raw)}
        dm    = {d: i for i,d in enumerate(dates)}

        cvli_raw  = np.zeros((N,T), np.float32)
        cvp_raw   = np.zeros((N,T), np.float32)
        intel_raw = np.zeros((N,T), np.float32)

        df_p = df[df["data"] >= START]
        df_tp= df_t[df_t["data"] >= START]

        for _,r in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvli_raw[ni,ti] = r["v"]
        for _,r in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvp_raw[ni,ti] = r["v"]
        for _,r in df_tp.groupby(["data","bairro"])["score_intel"].sum().reset_index().iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: intel_raw[ni,ti] = float(r["score_intel"])

        with open(self.latlon_file, encoding="utf-8") as f:
            raw_ll = json.load(f)
        ll_n   = {_norm(k): v for k,v in raw_ll.items()}
        coords = np.array([[ll_n.get(b,{"lat":0,"long":0})["lat"],
                            ll_n.get(b,{"lat":0,"long":0})["long"]]
                           for b in top_bairros_raw], np.float32)
        adj_mask = (cdist(coords,coords,"euclidean") < 0.05).astype(np.float32)
        np.fill_diagonal(adj_mask, 0)

        feats = {}

        cvli_cum = np.zeros((N,T), np.float32)
        cvp_cum  = np.zeros((N,T), np.float32)
        for ni in range(N):
            cvli_cum[ni] = pd.Series(cvli_raw[ni]).expanding().sum().values
            cvp_cum[ni]  = pd.Series(cvp_raw[ni]).expanding().sum().values
        hp_base = np.argsort(np.argsort(cvli_cum[:,-1])) / max(N-1,1)
        sqrt_hp = np.sqrt(np.clip(hp_base,0,1))[:,None]
        feats["cvp_cvli_ratio"] = (cvp_cum / (cvli_cum + 1)) * sqrt_hp

        te = np.zeros((N,T), np.float32)
        for ni in range(N):
            te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
        feats["target_enc"] = te

        for hl in [7,14,30]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(cvp_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"cvp_ewma_{hl}d"] = arr

        for hl in [7,14]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(intel_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"intel_ewma_{hl}d"] = arr

        r30 = np.zeros((N,T), np.float32)
        for ni in range(N):
            r30[ni] = pd.Series(cvli_raw[ni]).rolling(30,min_periods=1).sum().values
        nb = np.zeros((N,T), np.float32)
        for ti in range(T): nb[:,ti] = adj_mask @ r30[:,ti]
        feats["nbr_cvli_30d"] = nb

        hp = np.argsort(np.argsort(cvli_raw.sum(axis=1))) / max(N-1,1)
        feats["hist_pct"] = np.broadcast_to(hp[:,None],(N,T)).astype(np.float32).copy()

        e14 = np.zeros((N,T), np.float32)
        for ni in range(N):
            e14[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=14,min_periods=1).mean().values
        feats["inter_intel_cvli"] = feats["intel_ewma_7d"] * e14

        for hl in [7,14,30,60,90]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"cvli_ewma_{hl}d"] = arr

        # --- INTEGRAÇÃO CONTEXTUAL (V3.1 - PARIDADE 37 CANAIS) ---
        # Reconstruímos manualmente para evitar erros de versão do NumPy/Pickle
        try:
            from src.enrichment import is_brazil_holiday, is_cvp_hot_day
            weather_cache = {}
            weather_path = os.path.join(self.base_dir, "data", "weather_archive_cache.json")
            if os.path.exists(weather_path):
                with open(weather_path, 'r') as f:
                    weather_cache = json.load(f)
            
            f_canal = np.zeros((N, T), np.float32)
            h_canal = np.zeros((N, T), np.float32)
            c_canal = np.zeros((N, T), np.float32)
            
            for ti, d_val in enumerate(dates):
                is_h = 1.0 if is_brazil_holiday(d_val) else 0.0
                is_hot = 1.0 if is_cvp_hot_day(d_val) else 0.0
                precip = float(weather_cache.get(d_val.strftime('%Y-%m-%d'), 0.0))
                
                f_canal[:, ti] = is_h
                h_canal[:, ti] = is_hot
                c_canal[:, ti] = precip
                
            feats["feriado"]         = f_canal
            feats["dia_quente_cvp"]  = h_canal
            feats["chuva_mm"]        = c_canal

            # Interações para signal no LambdaRank
            hp_2d = feats["hist_pct"]
            feats["inter_chuva_hist"]   = c_canal * hp_2d
            feats["inter_feriado_hist"] = f_canal * hp_2d
        except Exception as e:
            print(f"⚠️ [CC] Falha ao injetar canais de contexto: {e}")

        return feats, dates, cvli_raw

    def _get_challenger_scores(self) -> dict | None:
        """Retorna dict {bairro_norm: score_normalizado} do LGBM."""
        try:
            T  = len(self._dates)
            N  = len(self._top_bairros)
            ti = T - 1

            xi = pd.DataFrame(
                [[float(self._feats[f][ni, ti]) for f in self._feat_names]
                 for ni in range(N)],
                columns=self._feat_names
            )
            sc_lgbm = self._ranker.predict(xi)

            sc_ewma = np.zeros(N, np.float32)
            for fn, w in self._ewma_weights.items():
                if fn in self._feats:
                    sc_ewma += w * self._feats[fn][:, ti]

            sc_ens = 0.5*_normalize_scores(sc_ewma) + 0.5*_normalize_scores(sc_lgbm)
            return {self._top_bairros[ni]: float(sc_ens[ni]) for ni in range(N)}
        except Exception as e:
            print(f"⚠️ [CC] Erro ao calcular challenger scores: {e}")
            return None

    def _evaluate_and_update(self, stgat_scores: dict):
        """
        Avalia P@10 de cada modelo contra CVLI real dos últimos EVAL_DAYS.
        Atualiza cc_weight via EMA e persiste decisão.
        """
        try:
            N  = len(self._top_bairros)
            T  = len(self._dates)

            # Ground truth: CVLI nos últimos EVAL_DAYS
            gt = self._cvli_raw[:, max(0, T-EVAL_DAYS):T].sum(axis=1)
            n_cvli = int(gt.sum())

            if n_cvli < MIN_CVLI_CC:
                self._log_decision("insufficient_signal", n_cvli, 0, 0, self._cc_weight)
                return

            # P@10 champion (ST-GAT)
            champ_arr = np.array([stgat_scores.get(b, 0.0) for b in self._top_bairros])
            p10_champ = _pk(champ_arr, gt, 10)

            # P@10 challenger (LGBM)
            chal_scores = self._get_challenger_scores()
            if chal_scores is None:
                self._log_decision("challenger_error", n_cvli, p10_champ, 0, self._cc_weight)
                return
            chal_arr = np.array([chal_scores.get(b, 0.0) for b in self._top_bairros])
            p10_chal = _pk(chal_arr, gt, 10)

            # Calcular novo peso via EMA
            advantage = p10_chal - p10_champ
            if advantage > MIN_ADVANTAGE:
                # Challenger melhor → aumentar peso gradualmente
                target_weight = min(MAX_CC_WEIGHT, self._cc_weight + advantage)
            elif advantage < -MIN_ADVANTAGE:
                # Champion melhor → diminuir peso
                target_weight = max(0.0, self._cc_weight + advantage)
            else:
                # Empate → manter
                target_weight = self._cc_weight

            # Suavização exponencial (evita oscilações bruscas)
            new_weight = ALPHA_EMA * target_weight + (1 - ALPHA_EMA) * self._cc_weight
            new_weight = float(np.clip(new_weight, 0.0, MAX_CC_WEIGHT))

            prev = self._cc_weight
            self._cc_weight = new_weight
            self._save_state()

            decision = "challenger_wins" if advantage > MIN_ADVANTAGE else (
                        "champion_wins"   if advantage < -MIN_ADVANTAGE else "tie")

            print(f"⚖️  [CC] n_cvli={n_cvli} | P@10 champion={p10_champ*100:.1f}% "
                  f"challenger={p10_chal*100:.1f}% | "
                  f"peso CC: {prev*100:.0f}%→{new_weight*100:.0f}% [{decision}]")

            self._log_decision(decision, n_cvli, p10_champ, p10_chal, new_weight)

        except Exception as e:
            print(f"⚠️ [CC] Erro na avaliação: {e}")

    def _log_decision(self, decision, n_cvli, p10_champ, p10_chal, cc_weight):
        """Persiste cada decisão do CC em logs/cc_decisions.jsonl."""
        os.makedirs(os.path.dirname(self.cc_log), exist_ok=True)
        entry = {
            "timestamp":    datetime.now().isoformat(),
            "decision":     decision,
            "n_cvli":       n_cvli,
            "p10_champion": round(float(p10_champ) * 100, 1),
            "p10_challenger": round(float(p10_chal) * 100, 1),
            "cc_weight":    round(float(cc_weight), 3),
            "champion_pct": round((1 - cc_weight) * 100, 1),
            "challenger_pct": round(cc_weight * 100, 1),
        }
        with open(self.cc_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
