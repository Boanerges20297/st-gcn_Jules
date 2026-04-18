"""
====================================================================
SENTINELA — MÓDULO DE INFERÊNCIA (Fase 4.2)
====================================================================
Interface limpa para carregar o modelo e gerar ranking de risco.

Uso direto:
  .\.venv\Scripts\python.exe tests/Sentinela/sentinela_inference.py

Como módulo (integração):
  from tests.Sentinela.sentinela_inference import SentinelaModel
  model = SentinelaModel()
  ranking = model.get_ranking()  # → List[dict]
====================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, warnings, unicodedata, pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

BASE_PATH   = r"c:\Users\Boanerges\Desktop\Projetos\Report Preview"
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

DATA_RAW    = os.path.join(BASE_PATH, "data", "raw")
MODEL_PATH  = os.path.join(BASE_PATH, "tests", "Sentinela", "lgbm_lean_v3_freeze.pkl")
OUT_PATH    = os.path.join(BASE_PATH, "tests", "Sentinela")

CSV_ENRICH  = os.path.join(DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
CSV_TROPA   = os.path.join(DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
LATLON_FILE = os.path.join(DATA_RAW, "bairros_centros_latlong.json")

PESO_NATUREZA = {
    'APREENSAO DE ARMA DE FOGO': 15.0, 'PORTE ILEGAL ART 14': 12.0,
    'TRAFICO DE DROGAS': 8.0, 'APREENSAO DE DROGAS': 6.0,
    'APREENSAO DE ENTORPECENTES': 6.0, 'MANDADO DE PRISAO': 4.0,
    'MANDADO EM ABERTO': 3.5, 'MANDADO DE PRISAO EM ABERTO': 3.5,
    'VEICULO ROUBADO RECUPERADO': 2.5, 'VEICULO ROUBADO LOCALIZADO': 2.0,
    'ABANDONO DE MATERIAL': 1.5, 'NAO INFORMADA': 0.5,
}


def norm(text):
    if pd.isna(text): return "DESCONHECIDO"
    t = unicodedata.normalize("NFD", str(text)).encode("ascii", "ignore").decode("utf-8")
    return t.strip().upper()

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9: return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────────
class SentinelaModel:
    """
    Interface de inferência do Sentinela V3.

    Exemplo de uso:
        model   = SentinelaModel()
        ranking = model.get_ranking()
        for r in ranking[:10]:
            print(r["rank"], r["bairro"], r["score"], r["razao_principal"])
    """

    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.ranker        = payload["ranker"]
        self.feat_names    = payload["feat_names_lgbm"]
        self.ewma_weights  = payload["ewma_weights"]
        self.top_bairros   = payload["top_bairros"]
        self.trained_at    = payload.get("trained_at", "desconhecido")
        self.dados_ate     = payload.get("dados_ate", "desconhecido")
        self.horizonte     = payload.get("horizonte_dias", 14)
        self._feats        = None
        self._dates        = None
        self._cvli_raw     = None

    # ── API pública ──────────────────────────────────────────────

    def get_ranking(self,
                    modo: str = "ensemble",
                    top_k: int = 40,
                    data_ref: date = None) -> list:
        """
        Retorna o ranking de risco dos bairros.

        Args:
            modo: "ensemble" (padrão) | "lgbm" | "ewma"
            top_k: número de bairros a retornar (1-40)
            data_ref: data de referência (padrão: último dia dos dados)

        Returns:
            Lista de dicts com keys:
              rank, bairro, score, score_lgbm, score_ewma,
              cvp_ratio, intel_14d, nbr_30d, hist_pct,
              razao_principal, previsao_inicio, previsao_fim
        """
        if self._feats is None:
            self._build_features()

        dates  = self._dates
        ti     = self._get_ti(data_ref, dates)
        N      = len(self.top_bairros)

        # Scores LGBM
        xi = pd.DataFrame(
            [[float(self._feats[f][ni, ti]) for f in self.feat_names]
             for ni in range(N)],
            columns=self.feat_names
        )
        sc_lgbm = self.ranker.predict(xi)

        # Scores EWMA-Multi
        sc_ewma = np.zeros(N, np.float32)
        for fname, w in self.ewma_weights.items():
            if fname in self._feats:
                sc_ewma += w * self._feats[fname][:, ti]

        # Ensemble
        sc_ens = (0.5 * normalize_scores(sc_ewma)
                + 0.5 * normalize_scores(sc_lgbm))

        scores_map = {"ensemble": sc_ens, "lgbm": sc_lgbm, "ewma": sc_ewma}
        sc = scores_map.get(modo, sc_ens)

        # Feature importance para razão principal
        fi = self.ranker.feature_importances_

        previsao_inicio = dates[ti].date()
        previsao_fim    = previsao_inicio + timedelta(days=self.horizonte)

        ranking = []
        for rank_i, ni in enumerate(np.argsort(sc)[::-1][:top_k], 1):
            feat_vals = {f: float(self._feats[f][ni, ti]) for f in self.feat_names}
            # Razão principal = feature mais importante com valor acima da mediana
            razao = self._razao_principal(feat_vals, fi)
            ranking.append({
                "rank":             rank_i,
                "bairro":           self.top_bairros[ni],
                "score":            round(float(normalize_scores(sc)[ni]), 4),
                "score_lgbm":       round(float(normalize_scores(sc_lgbm)[ni]), 4),
                "score_ewma":       round(float(normalize_scores(sc_ewma)[ni]), 4),
                "cvp_ratio":        round(feat_vals.get("cvp_cvli_ratio", 0), 3),
                "intel_14d":        round(feat_vals.get("intel_ewma_14d", 0), 2),
                "nbr_30d":          round(feat_vals.get("nbr_cvli_30d", 0), 2),
                "hist_pct":         round(feat_vals.get("hist_pct", 0), 3),
                "razao_principal":  razao,
                "previsao_inicio":  str(previsao_inicio),
                "previsao_fim":     str(previsao_fim),
            })
        return ranking

    def get_alerts(self, limiar_intel: float = 0.5) -> list:
        """
        Retorna bairros com intel de tropa recente acima do limiar,
        mesmo que não estejam no top-10 de risco geral.
        Útil para flagrar emergências táticas.
        """
        if self._feats is None:
            self._build_features()
        ti = len(self._dates) - 1
        N  = len(self.top_bairros)
        alerts = []
        for ni in range(N):
            intel = float(self._feats["intel_ewma_14d"][ni, ti])
            if intel >= limiar_intel:
                alerts.append({
                    "bairro":       self.top_bairros[ni],
                    "intel_14d":    round(intel, 2),
                    "intel_7d":     round(float(self._feats["intel_ewma_7d"][ni, ti]), 2),
                    "alerta":       "🚨 ALTO" if intel >= 2.0 else "⚠️ MODERADO",
                })
        return sorted(alerts, key=lambda x: -x["intel_14d"])

    def summary(self) -> str:
        """Retorna string com resumo do modelo."""
        return (
            f"SentinelaModel V3 Freeze\n"
            f"  Treinado em: {self.trained_at}\n"
            f"  Dados até:   {self.dados_ate}\n"
            f"  Bairros:     {len(self.top_bairros)}\n"
            f"  Features:    {len(self.feat_names)}\n"
            f"  Horizonte:   {self.horizonte} dias"
        )

    # ── Internos ─────────────────────────────────────────────────

    def _get_ti(self, data_ref, dates):
        if data_ref is None:
            return len(dates) - 1
        ts = pd.Timestamp(data_ref)
        idx = np.searchsorted([d.value for d in dates], ts.value)
        return min(idx, len(dates) - 1)

    def _razao_principal(self, feat_vals: dict, fi: np.ndarray) -> str:
        """
        Identifica POR QUE este bairro está no ranking.
        Calcula o z-score de cada feature em relação ao conjunto
        dos 40 bairros e retorna a mais destacada ponderada por importância.
        """
        LABELS = {
            "cvp_cvli_ratio":  "CVP escalando para CVLI",
            "target_enc":      "Histórico de CVLI elevado",
            "intel_ewma_14d":  "Intel de tropa recente",
            "cvp_ewma_30d":    "Tendência de CVP longa",
            "inter_intel_cvli":"Pressão Intel + CVLI simultâneos",
            "nbr_cvli_30d":    "CVLI em bairros vizinhos",
            "intel_ewma_7d":   "Intel de tropa (curto prazo)",
            "hist_pct":        "Bairro estruturalmente perigoso",
            "cvp_ewma_14d":    "Tendência de CVP média",
            "cvp_ewma_7d":     "Tendência de CVP curta",
        }
        if self._feats is None or self._dates is None:
            return "Padrão histórico"
        ti = len(self._dates) - 1
        N  = len(self.top_bairros)
        # Calcula z-score de cada feature para o bairro atual vs todos
        best_score = -999.0
        best_label = "Padrão histórico"
        for j, fname in enumerate(self.feat_names):
            vals_all = np.array([float(self._feats[fname][ni, ti]) for ni in range(N)])
            std = vals_all.std()
            if std < 1e-6: continue
            z = (feat_vals.get(fname, 0) - vals_all.mean()) / std
            weighted = z * (fi[j] / fi.sum())  # pondera pela importância
            if weighted > best_score and z > 0.3:  # só destaque positivo real
                best_score = weighted
                best_label = LABELS.get(fname, fname)
        return best_label

    def _build_features(self):
        """Constrói a matriz de features a partir dos CSVs brutos."""
        START = pd.Timestamp("2024-01-01")

        df = pd.read_csv(CSV_ENRICH, low_memory=False)
        df = df[df["cidade"].str.upper() == "FORTALEZA"].copy()
        df["bairro"] = df["bairro"].apply(norm)
        df["data"]   = pd.to_datetime(df["data"], errors="coerce")
        df           = df.dropna(subset=["data", "bairro"])
        df["is_cvli"]= (df["tipo"] == "cvli").astype("int8")
        df["is_cvp"] = (df["tipo"] == "cvp").astype("int8")

        df_t = pd.read_csv(CSV_TROPA, low_memory=False, encoding="utf-8-sig")
        df_t["bairro"]     = df_t["bairro"].apply(norm)
        df_t["data"]       = pd.to_datetime(df_t["data"], errors="coerce")
        df_t               = df_t.dropna(subset=["data","bairro"])
        df_t["peso_nat"]   = df_t["natureza"].str.upper().str.strip().map(
            lambda x: PESO_NATUREZA.get(x, 1.0))
        df_t["score_intel"]= (
            df_t["qtd_armas"]*15 + np.log1p(df_t["qtd_drogas"].fillna(0))*4
            + df_t["qtd_drogas_itens"]*2 + df_t["qtd_veiculos_apreendidos"]*3
            + df_t["peso_nat"]
        ).astype("float32")

        top_bairros = self.top_bairros
        dates   = pd.date_range(START, df["data"].max(), freq="D")
        N, T    = len(top_bairros), len(dates)
        nm      = {b: i for i, b in enumerate(top_bairros)}
        dm      = {d: i for i, d in enumerate(dates)}

        cvli_raw  = np.zeros((N,T), np.float32)
        cvp_raw   = np.zeros((N,T), np.float32)
        intel_raw = np.zeros((N,T), np.float32)

        df_p  = df[df["data"] >= START]
        df_tp = df_t[df_t["data"] >= START]

        for _,r in df_p[df_p["is_cvli"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvli_raw[ni,ti] = r["v"]

        for _,r in df_p[df_p["is_cvp"]==1].groupby(["data","bairro"]).size().reset_index(name="v").iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: cvp_raw[ni,ti] = r["v"]

        for _,r in df_tp.groupby(["data","bairro"])["score_intel"].sum().reset_index().iterrows():
            ni,ti = nm.get(r["bairro"]), dm.get(r["data"])
            if ni is not None and ti is not None: intel_raw[ni,ti] = float(r["score_intel"])

        with open(LATLON_FILE, encoding="utf-8") as f:
            raw_ll = json.load(f)
        ll_n     = {norm(k): v for k,v in raw_ll.items()}
        coords   = np.array([[ll_n.get(b,{"lat":0,"long":0})["lat"],
                              ll_n.get(b,{"lat":0,"long":0})["long"]]
                             for b in top_bairros], np.float32)
        adj_mask = (cdist(coords,coords,"euclidean") < 0.05).astype(np.float32)
        np.fill_diagonal(adj_mask, 0)

        feats = {}

        # cvp_cvli_ratio calibrado
        cvli_cum = np.zeros((N,T), np.float32)
        cvp_cum  = np.zeros((N,T), np.float32)
        for ni in range(N):
            cvli_cum[ni] = pd.Series(cvli_raw[ni]).expanding().sum().values
            cvp_cum[ni]  = pd.Series(cvp_raw[ni]).expanding().sum().values
        hist_pct_base = np.argsort(np.argsort(cvli_cum[:,-1])) / max(N-1,1)
        sqrt_hp = np.sqrt(np.clip(hist_pct_base, 0, 1))[:,None]
        feats["cvp_cvli_ratio"] = (cvp_cum / (cvli_cum + 1)) * sqrt_hp

        # target_enc
        te = np.zeros((N,T), np.float32)
        for ni in range(N):
            te[ni] = pd.Series(cvli_raw[ni]).expanding().mean().values
        feats["target_enc"] = te

        # cvp_ewma
        for hl in [7,14,30]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(cvp_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"cvp_ewma_{hl}d"] = arr

        # intel_ewma
        for hl in [7,14]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(intel_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"intel_ewma_{hl}d"] = arr

        # nbr_cvli_30d
        r30 = np.zeros((N,T), np.float32)
        for ni in range(N):
            r30[ni] = pd.Series(cvli_raw[ni]).rolling(30,min_periods=1).sum().values
        nb = np.zeros((N,T), np.float32)
        for ti in range(T): nb[:,ti] = adj_mask @ r30[:,ti]
        feats["nbr_cvli_30d"] = nb

        # hist_pct
        hp = np.argsort(np.argsort(cvli_raw.sum(axis=1))) / max(N-1,1)
        feats["hist_pct"] = np.broadcast_to(hp[:,None],(N,T)).astype(np.float32).copy()

        # inter_intel_cvli
        e14 = np.zeros((N,T), np.float32)
        for ni in range(N):
            e14[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=14,min_periods=1).mean().values
        feats["inter_intel_cvli"] = feats["intel_ewma_7d"] * e14

        # cvli_ewma para EWMA-Multi
        for hl in [7,14,30,60,90]:
            arr = np.zeros((N,T), np.float32)
            for ni in range(N):
                arr[ni] = pd.Series(cvli_raw[ni]).ewm(halflife=hl,min_periods=1).mean().values
            feats[f"cvli_ewma_{hl}d"] = arr

        # --- INTEGRAÇÃO CONTEXTUAL (V3.1 - PARIDADE 37 CANAIS) ---
        try:
            from src.enrichment import is_brazil_holiday, is_cvp_hot_day
            weather_cache = {}
            weather_path = os.path.join(BASE_PATH, "data", "weather_archive_cache.json")
            if os.path.exists(weather_path):
                with open(weather_path, 'r', encoding='utf-8') as f:
                    weather_cache = json.load(f)
            
            f_canal = np.zeros((N, T), np.float32)
            h_canal = np.zeros((N, T), np.float32)
            c_canal = np.zeros((N, T), np.float32)
            
            for ti, d_val in enumerate(dates):
                is_h = 1.0 if is_brazil_holiday(d_val) else 0.0
                is_hot = 1.0 if is_cvp_hot_day(d_val) else 0.0
                # Tenta buscar no cache ou via função real
                d_str = d_val.strftime('%Y-%m-%d')
                precip = float(weather_cache.get(d_str, 0.0))
                
                f_canal[:, ti] = is_h
                h_canal[:, ti] = is_hot
                c_canal[:, ti] = precip
                
            feats["feriado"]         = f_canal
            feats["dia_quente_cvp"]  = h_canal
            feats["chuva_mm"]        = c_canal

            # Interações para signal no LambdaRank (V3.1 Elite)
            hp_2d = feats["hist_pct"]
            feats["inter_chuva_hist"]   = c_canal * hp_2d
            feats["inter_feriado_hist"] = f_canal * hp_2d
            print(f"✅ Contexto injetado: Clima({c_canal.sum():.1f}mm) | Feriados({f_canal.sum():.0f})")
        except Exception as e:
            print(f"⚠️ [CC] Falha ao injetar canais de contexto: {e}")

        self._feats    = feats
        self._dates    = dates
        self._cvli_raw = cvli_raw


# ─────────────────────────────────────────────────────────────────
#  Execução direta
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*68)
    print("  SENTINELA — RANKING DE RISCO ATUAL")
    print("="*68)

    model = SentinelaModel()
    print(f"\n{model.summary()}\n")

    ranking = model.get_ranking(modo="ensemble", top_k=40)

    # Top-20
    print(f"  {'Rank':<5} {'Bairro':<28} {'Score':>7}  {'Razão Principal'}")
    print("  " + "─"*75)
    for r in ranking[:20]:
        print(f"  {r['rank']:<5} {r['bairro']:<28} {r['score']:>7.4f}  {r['razao_principal']}")

    print(f"\n  Previsão: {ranking[0]['previsao_inicio']} → {ranking[0]['previsao_fim']}")

    # Alertas de intel
    alerts = model.get_alerts(limiar_intel=0.5)
    if alerts:
        print(f"\n  🚨 ALERTAS DE INTEL DE TROPA:")
        for a in alerts:
            print(f"     {a['alerta']}  {a['bairro']:<28} intel_14d={a['intel_14d']:.2f}")

    # Salvar JSON para integração
    out_json = os.path.join(OUT_PATH, "ranking_sentinela_atual.json")
    import json as jsonlib
    with open(out_json, "w", encoding="utf-8") as f:
        jsonlib.dump({
            "gerado_em": str(datetime.now()),
            "previsao_inicio": ranking[0]["previsao_inicio"],
            "previsao_fim": ranking[0]["previsao_fim"],
            "ranking": ranking,
            "alertas_intel": alerts,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] JSON exportado: {out_json}")
    print("="*68)


if __name__ == "__main__":
    main()
