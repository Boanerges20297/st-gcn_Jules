"""
====================================================================
SENTINELA — SCRIPT DE PROMOÇÃO (Fase 4.1)
====================================================================
Promove o modelo candidato de tests/Sentinela/ para models/active/
com checklist de segurança, backup do modelo anterior e registro em log.

Uso:
  .\.venv\Scripts\python.exe tests/Sentinela/promote_model.py
  .\.venv\Scripts\python.exe tests/Sentinela/promote_model.py --force
====================================================================
"""

import sys, io, argparse, shutil, json, os
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TESTS_DIR    = os.path.join(BASE_PATH, "tests", "Sentinela")
ACTIVE_DIR   = os.path.join(BASE_PATH, "models", "active")
ARCHIVE_DIR  = os.path.join(BASE_PATH, "models", "archive")
PROMO_LOG    = os.path.join(BASE_PATH, "models", "promotion_log.json")

CANDIDATE    = os.path.join(TESTS_DIR, "lgbm_lean_v3_freeze.pkl")
REPORT       = os.path.join(TESTS_DIR, "freeze_report.txt")
RANKING_CSV  = os.path.join(TESTS_DIR, "ranking_atual_v3_freeze.csv")
RANKING_JSON = os.path.join(TESTS_DIR, "ranking_sentinela_atual.json")

# ─────────────────────────────────────────────────────────────────
CHECKLIST = [
    ("Modelo candidato existe",           lambda: os.path.exists(CANDIDATE)),
    ("Relatório de treino existe",        lambda: os.path.exists(REPORT)),
    ("Ranking atual gerado",              lambda: os.path.exists(RANKING_CSV)),
    ("Modelo >= 500 KB (não corrompido)", lambda: os.path.getsize(CANDIDATE) >= 500_000),
]

# ─────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def run(force=False):
    section("SENTINELA — PROMOÇÃO DE MODELO (Fase 4.1)")
    print(f"\n  Data:      {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Candidato: {CANDIDATE}")
    print(f"  Destino:   {ACTIVE_DIR}\n")

    # ── 1. Checklist automático ──
    section("1. CHECKLIST AUTOMÁTICO")
    falhas = []
    for nome, check_fn in CHECKLIST:
        ok = check_fn()
        status = "✅" if ok else "❌"
        print(f"  {status}  {nome}")
        if not ok:
            falhas.append(nome)

    if falhas:
        print(f"\n  ❌ {len(falhas)} verificação(ões) falharam. Abortando promoção.")
        for f in falhas: print(f"     - {f}")
        return False

    # ── 2. Checklist manual (interativo) ──
    if not force:
        section("2. CHECKLIST MANUAL")
        perguntas = [
            "Os top-10 bairros fazem sentido operacionalmente?",
            "Não há bairros comerciais/sem histórico de CVLI no top-5?",
            "Os alertas de Intel estão sendo investigados no campo?",
            "A equipe está ciente das previsões para os próximos 14 dias?",
        ]
        print()
        for i, p in enumerate(perguntas, 1):
            resp = input(f"  [{i}] {p} (s/n): ").strip().lower()
            if resp != "s":
                print(f"\n  ❌ Promoção cancelada pelo operador na questão {i}.")
                return False
    else:
        print("\n  ⚠️  Modo --force: checklist manual ignorado.")

    # ── 3. Backup do modelo atual (se existir) ──
    section("3. BACKUP DO MODELO ANTERIOR")
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    modelos_ativos = [f for f in os.listdir(ACTIVE_DIR) if f.endswith(".pkl")]
    if modelos_ativos:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for m in modelos_ativos:
            src = os.path.join(ACTIVE_DIR, m)
            dst = os.path.join(ARCHIVE_DIR, f"{ts}_{m}")
            shutil.copy2(src, dst)
            print(f"  [BACKUP] {m} → archive/{ts}_{m}")
    else:
        print("  [INFO] Nenhum modelo ativo anterior encontrado.")

    # ── 4. Promoção ──
    section("4. PROMOVENDO MODELO")
    dest_pkl  = os.path.join(ACTIVE_DIR, "lgbm_lean_v3_freeze.pkl")
    dest_csv  = os.path.join(ACTIVE_DIR, "ranking_atual.csv")
    dest_json = os.path.join(ACTIVE_DIR, "ranking_atual.json")
    dest_rep  = os.path.join(ACTIVE_DIR, "modelo_report.txt")

    shutil.copy2(CANDIDATE,   dest_pkl)
    shutil.copy2(RANKING_CSV, dest_csv)
    if os.path.exists(RANKING_JSON): shutil.copy2(RANKING_JSON, dest_json)
    if os.path.exists(REPORT):       shutil.copy2(REPORT,       dest_rep)

    print(f"  [OK] Modelo:   {dest_pkl}")
    print(f"  [OK] Ranking:  {dest_csv}")
    print(f"  [OK] JSON:     {dest_json}")
    print(f"  [OK] Relatório:{dest_rep}")

    # ── 5. Registrar no promotion_log.json ──
    section("5. REGISTRANDO NO LOG")
    import pickle
    with open(dest_pkl, "rb") as f:
        meta = pickle.load(f)

    entry = {
        "timestamp":    datetime.now().isoformat(),
        "modelo":       "lgbm_lean_v3_freeze",
        "versao":       meta.get("versao", "v3_freeze"),
        "dados_ate":    meta.get("dados_ate", "?"),
        "treinado_em":  meta.get("trained_at", "?"),
        "bairros":      len(meta.get("top_bairros", [])),
        "features":     len(meta.get("feat_names_lgbm", [])),
        "performance":  {
            "shadow_ewma_p10": "50.0%",
            "shadow_lgbm_p20": "70.0%",
            "shadow_periodo":  "31/Mar→13/Abr/2026"
        },
        "promovido_por": "operador" if not force else "script --force",
        "backup_anterior": [f for f in os.listdir(ARCHIVE_DIR)] if os.path.exists(ARCHIVE_DIR) else [],
    }

    log = []
    if os.path.exists(PROMO_LOG):
        with open(PROMO_LOG, encoding="utf-8") as f:
            try: log = json.load(f)
            except: log = []

    log.append(entry)
    with open(PROMO_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Log: {PROMO_LOG}")

    # ── Sumário ──
    section("✅ PROMOÇÃO CONCLUÍDA")
    print(f"\n  Modelo:      lgbm_lean_v3_freeze.pkl")
    print(f"  Dados até:   {meta.get('dados_ate','?')}")
    print(f"  Localização: {ACTIVE_DIR}")
    print(f"  Previsão:    próximos 14 dias a partir de {meta.get('dados_ate','?')}")
    print(f"\n  ⚠️  Lembre-se: re-treinar ao receber novos dados:")
    print(f"     .\.venv\Scripts\python.exe tests/Sentinela/freeze_total_v3.py")
    print(f"     .\.venv\Scripts\python.exe tests/Sentinela/promote_model.py")
    print("="*60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promoção do modelo Sentinela para produção")
    parser.add_argument("--force", action="store_true",
                        help="Pula checklist manual interativo (para uso em automação)")
    args = parser.parse_args()
    ok = run(force=args.force)
    sys.exit(0 if ok else 1)
