import io
import json
import os
import sys
import importlib.util

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FREEZE_SCRIPT = os.path.join(os.path.dirname(__file__), "freeze_total_v3.py")
RANKINGS_JSON = os.path.join(BASE_PATH, "outputs", "cvli_first_candidate_rankings.json")
METADATA_JSON = os.path.join(BASE_PATH, "outputs", "cvli_first_candidate_metadata.json")
OUT_JSON = os.path.join(BASE_PATH, "outputs", "cvli_first_candidate_validation.json")


def load_freeze_module():
    spec = importlib.util.spec_from_file_location("freeze_total_v3_local", FREEZE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", closefd=False)
    module.sys.stdout = sys.stdout
    module.sys.stderr = sys.stderr
    return module


def evaluate():
    freeze = load_freeze_module()
    _, _, dates, cvli_raw, top_bairros = freeze.build_all()

    with open(RANKINGS_JSON, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    with open(METADATA_JSON, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    ref_date = pd.Timestamp(meta["reference_date"])
    pred_end = pd.Timestamp(meta["prediction_end"])
    date_map = {pd.Timestamp(date): index for index, date in enumerate(dates)}

    if pred_end > pd.Timestamp(dates[-1]):
        print("Ainda não há dados suficientes para validar toda a janela prevista.")
        print(f"Último dado disponível: {pd.Timestamp(dates[-1]).date()} | Fim da previsão: {pred_end.date()}")
        return

    start_i = date_map[ref_date]
    end_i = date_map[pred_end]
    gt = cvli_raw[:, start_i + 1 : end_i + 1].sum(axis=1)
    gt_map = {top_bairros[i]: float(gt[i]) for i in range(len(top_bairros))}

    rankings = {}
    for row in rows:
        rankings.setdefault(row["modelo"], []).append(row)

    results = {}
    for model_name, model_rows in rankings.items():
        ordered = sorted(model_rows, key=lambda item: item["rank"])
        top10 = [item["bairro"] for item in ordered[:10]]
        top20 = [item["bairro"] for item in ordered[:20]]
        hits10 = sum(1 for bairro in top10 if gt_map.get(bairro, 0) > 0)
        hits20 = sum(1 for bairro in top20 if gt_map.get(bairro, 0) > 0)
        total_pos = sum(1 for value in gt_map.values() if value > 0)
        results[model_name] = {
            "p10": hits10 / 10,
            "p20": hits20 / 20,
            "r10": hits10 / max(total_pos, 1),
            "r20": hits20 / max(total_pos, 1),
            "hits10": hits10,
            "hits20": hits20,
            "total_positive_bairros": total_pos,
        }

    payload = {
        "validated_at": pd.Timestamp.now().isoformat(),
        "reference_date": meta["reference_date"],
        "prediction_end": meta["prediction_end"],
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n=== Validação dos candidatos CVLI-first ===")
    for model_name, values in results.items():
        print(
            f"{model_name}: "
            f"P@10={values['p10']*100:.1f}% "
            f"P@20={values['p20']*100:.1f}% "
            f"R@10={values['r10']*100:.1f}% "
            f"R@20={values['r20']*100:.1f}%"
        )
    print(f"\n[OK] {OUT_JSON}")


if __name__ == "__main__":
    evaluate()
