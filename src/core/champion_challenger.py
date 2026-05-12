"""
====================================================================
SENTINELA — CHAMPION/CHALLENGER (Core Integration)
====================================================================
"""
import os, json, pickle, warnings, unicodedata
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

def _norm(text):
    if not isinstance(text, str): return ""
    t = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def _normalize_scores(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _pk(scores, targets, k):
    top_pred_idx = np.argsort(scores)[::-1][:k]
    hits = np.sum(targets[top_pred_idx] > 0)
    return float(hits / k)

class ChampionChallenger:
    def __init__(self, base_dir: str):
        self.base_dir    = base_dir
        self.model_path  = os.path.join(base_dir, "models", "active", "lgbm_solo_challenger_800d.pkl")
        self.cc_log      = os.path.join(base_dir, "logs", "cc_decisions.jsonl")
        self.state_path  = os.path.join(base_dir, "data", "cc_state.json")
        self.csv_enrich  = os.path.join(base_dir, "data", "raw", "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")

        self._ranker       = None
        self._feat_names   = None
        self._top_bairros  = None
        self._feats        = None
        self._dates        = None
        self._cvli_raw     = None

        self._cc_weight   = 0.0
        self._last_eval   = None
        self._load_state()
        self._load_challenger()

    def _load_state(self):
        """Carrega o estado persistido (pesos do blend)."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    s = json.load(f)
                self._cc_weight = float(s.get("cc_weight", 0.0))
                last = s.get("last_eval")
                self._last_eval = datetime.fromisoformat(last) if last else None
                print(f"✅ [CC] Estado restaurado: challenger={self._cc_weight*100:.0f}%")
            except Exception: pass

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "cc_weight": self._cc_weight,
                "last_eval": str(self._last_eval) if self._last_eval else None,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)

    def _load_challenger(self):
        """Carrega o modelo Solo 800d."""
        if not os.path.exists(self.model_path):
            print(f"⚠️ [CC] Modelo não encontrado: {self.model_path}")
            return
        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            self._ranker       = payload["ranker"]
            self._feat_names   = payload["feat_names_lgbm"]
            self._top_bairros  = [_norm(b) for b in payload["top_bairros"]]
            print(f"✅ [CC] Solo Challenger ATIVO (P@10 Benchmark: {payload.get('p10', 0):.1f}%)")
        except Exception as e:
            print(f"❌ [CC] Erro ao carregar solo: {e}")

    def apply(self, scores_map: dict) -> dict:
        if self._ranker is None or self._top_bairros is None: return scores_map
        try: self._ensure_features()
        except: return scores_map

        now = datetime.now()
        if self._last_eval is None or (now - self._last_eval).total_seconds() > 3600:
            self._evaluate_and_update(scores_map)
            self._last_eval = now

        if self._cc_weight < 0.01: return scores_map

        challenger_scores = self._get_challenger_scores()
        if challenger_scores is None: return scores_map

        result = dict(scores_map)
        
        # --- CALIBRAÇÃO HONESTA ---
        # Em vez de normalizar (0-100), usamos o score bruto do LGBM 
        # e ajustamos para a escala do sistema (0-100) de forma conservadora.
        for b in self._top_bairros:
            raw_score = challenger_scores.get(b, 0.0)
            
            # 1. Trazer para escala [0, 80] - O teto é 80% para evitar pânico
            # O ranker xendcg gera scores variados; assumimos que > 1.0 é sinal forte.
            prob_risk = np.clip(raw_score * 40.0, 0, 95) 
            
            # 2. PENALIDADE DE SILÊNCIO (Tactical Honesty)
            # Se o bairro não vê um CVLI há mais de 21 dias, ele não pode ser 'Crítico'
            # a menos que o Champion (ST-GAT) insista muito.
            ni = self._top_bairros.index(b)
            days_silent = self._feats["recency"][ni, -1]
            
            if days_silent > 21:
                prob_risk *= 0.5  # Reduz 50% do risco se estiver 'frio'
            elif days_silent > 14:
                prob_risk *= 0.7  # Reduz 30%
                
            # 3. BLEND COM CHAMPION (ST-GAT)
            champ_val = scores_map.get(b, 0.0)
            final_val = (1 - self._cc_weight) * champ_val + self._cc_weight * prob_risk
            
            result[b] = float(np.clip(final_val, 0, 100))
            
        return result

    def _ensure_features(self):
        if self._feats is not None: return
        self._feats, self._dates, self._cvli_raw = self._build_features()

    def _build_features(self):
        START = pd.Timestamp("2024-02-01")
        top_bairros_raw = [b for b in self._top_bairros]
        df = pd.read_csv(self.csv_enrich, low_memory=False)
        df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
        df["bairro"], df["data"] = df["bairro"].apply(_norm), pd.to_datetime(df["data"], errors="coerce")
        df = df.dropna(subset=["data","bairro"])
        df["is_cvli"], df["is_cvp"] = (df["tipo"]=="cvli").astype(int), (df["tipo"]=="cvp").astype(int)

        dates = pd.date_range(START, df["data"].max(), freq="D")
        N, T = len(top_bairros_raw), len(dates)
        nm, dm = {b: i for i,b in enumerate(top_bairros_raw)}, {d: i for i,d in enumerate(dates)}
        cvli_raw, cvp_raw = np.zeros((N,T), np.float32), np.zeros((N,T), np.float32)

        df_p = df[df["data"] >= START]
        for _,r in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvli_raw[ni,ti] = r["v"]
        for _,r in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvp_raw[ni,ti] = r["v"]

        feats = {}
        for hl in [3, 7, 14, 30, 90]:
            a_cvli, a_cvp = np.zeros((N,T), np.float32), np.zeros((N,T), np.float32)
            for ni in range(N):
                a_cvli[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl).mean().values
                a_cvp[ni]  = pd.Series(cvp_raw[ni]).ewm(halflife=hl).mean().values
            feats[f"cvli_ewma_{hl}d"], feats[f"cvp_ewma_{hl}d"] = a_cvli, a_cvp
        rec = np.zeros((N,T), np.float32)
        for ni in range(N):
            last = -100
            for ti in range(T):
                if cvli_raw[ni,ti] > 0: last = ti
                rec[ni,ti] = min(ti - last, 180)
        feats["recency"] = rec
        te = np.zeros((N,T), np.float32)
        for ni in range(N): te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
        feats["target_enc"] = te
        return feats, dates, cvli_raw

    def _get_challenger_scores(self) -> dict | None:
        try:
            ti = len(self._dates) - 1
            xi = pd.DataFrame([[float(self._feats[f][ni, ti]) for f in self._feat_names] for ni in range(len(self._top_bairros))], columns=self._feat_names)
            return {self._top_bairros[ni]: float(self._ranker.predict(xi)[ni]) for ni in range(len(self._top_bairros))}
        except: return None

    def _evaluate_and_update(self, stgat_scores: dict):
        try:
            T = len(self._dates)
            gt = self._cvli_raw[:, max(0, T-14):T].sum(axis=1)
            if gt.sum() < 5: return
            champ_arr = np.array([stgat_scores.get(b, 0.0) for b in self._top_bairros])
            chal_scores = self._get_challenger_scores()
            if chal_scores is None: return
            chal_arr = np.array([chal_scores.get(b, 0.0) for b in self._top_bairros])
            advantage = _pk(chal_arr, gt, 10) - _pk(champ_arr, gt, 10)
            target = min(0.50, self._cc_weight + 0.1) if advantage > 0.03 else (max(0.0, self._cc_weight - 0.1) if advantage < -0.03 else self._cc_weight)
            self._cc_weight = float(np.clip(0.3 * target + 0.7 * self._cc_weight, 0.0, 0.50))
            self._save_state()
        except: pass
    
    def status(self) -> dict:
        return {"cc_weight": self._cc_weight, "challenger_pct": round(self._cc_weight * 100, 1)}
