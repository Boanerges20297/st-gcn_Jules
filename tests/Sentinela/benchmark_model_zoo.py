import io
import os
import sys
import time
import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_correto.py")
OUT_DIR = os.path.join(os.path.dirname(__file__))


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_correto_local", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", closefd=False)
    module.sys.stdout = sys.stdout
    module.sys.stderr = sys.stderr
    module.BASE_PATH = BASE_PATH
    module.DATA_RAW = os.path.join(BASE_PATH, "data", "raw")
    module.DATA_PROC = os.path.join(BASE_PATH, "data", "processed")
    module.MODELS_PATH = os.path.join(BASE_PATH, "models", "active")
    module.OUT_PATH = os.path.join(BASE_PATH, "tests", "Sentinela")
    module.CKPT_PATH = os.path.join(module.MODELS_PATH, "fortaleza_model_active.pth")
    module.CSV_ENRICH = os.path.join(module.DATA_RAW, "dados_status_ocorrencias_gerais_ENRIQUECIDO.csv")
    module.CSV_TROPA = os.path.join(module.DATA_RAW, "ocorrencias_tropa_limpo_fortaleza.csv")
    module.LATLON_FILE = os.path.join(module.DATA_RAW, "bairros_centros_latlong.json")
    return module


def build_flat_features(data, idx_range, horizon, is_train=True, step=1):
    feat = data["node_features"]
    num_nodes, total_days, _ = feat.shape
    rows = []
    for time_index in list(idx_range)[::step]:
        if time_index < 60 or time_index + horizon >= total_days:
            continue
        targets_h = feat[:, time_index + 1 : time_index + horizon + 1, 0].sum(axis=1)
        if targets_h.sum() == 0 and is_train:
            continue

        for node_index in range(num_nodes):
            row_feat = []
            for halflife in [3, 7, 14, 30, 60]:
                series = pd.Series(feat[node_index, max(0, time_index - 60) : time_index, 0])
                row_feat.append(float(series.ewm(halflife=halflife).mean().iloc[-1]) if len(series) > 0 else 0.0)
            for halflife in [7, 14, 30]:
                series = pd.Series(feat[node_index, max(0, time_index - 30) : time_index, 1])
                row_feat.append(float(series.ewm(halflife=halflife).mean().iloc[-1]) if len(series) > 0 else 0.0)
            for halflife in [7, 14]:
                series = pd.Series(feat[node_index, max(0, time_index - 30) : time_index, 27])
                row_feat.append(float(series.ewm(halflife=halflife).mean().iloc[-1]) if len(series) > 0 else 0.0)

            row_feat += [
                float(feat[node_index, max(0, time_index - 7) : time_index, 0].sum()),
                float(feat[node_index, max(0, time_index - 14) : time_index, 0].sum()),
                float(feat[node_index, max(0, time_index - 30) : time_index, 0].sum()),
            ]
            row_feat += list(feat[node_index, time_index, 3:23])
            row_feat.append(float(node_index))

            label = int(targets_h[node_index] > 0)
            rows.append(
                {
                    "ti": time_index,
                    "ni": node_index,
                    "label": label,
                    **{f"f{feature_index}": value for feature_index, value in enumerate(row_feat)},
                }
            )
    return pd.DataFrame(rows)


def tabular_scores(data, train_range, test_range, model_builder, horizon):
    num_nodes = data["node_features"].shape[0]
    df_train = build_flat_features(data, train_range, horizon, is_train=True, step=3)
    df_test = build_flat_features(data, test_range, horizon, is_train=False, step=1)
    feature_cols = [column for column in df_train.columns if column.startswith("f")]

    if df_train.empty or df_test.empty:
        return np.zeros((num_nodes, len(list(test_range))), dtype=np.float32)

    model = model_builder()
    model.fit(df_train[feature_cols], df_train["label"].astype("int32"))

    if hasattr(model, "predict_proba"):
        df_test["score"] = model.predict_proba(df_test[feature_cols])[:, 1]
    elif hasattr(model, "decision_function"):
        df_test["score"] = model.decision_function(df_test[feature_cols])
    else:
        df_test["score"] = model.predict(df_test[feature_cols])

    scores = np.zeros((num_nodes, len(list(test_range))), dtype=np.float32)
    time_to_col = {time_index: col for col, time_index in enumerate(list(test_range))}
    for _, row in df_test.iterrows():
        col = time_to_col.get(int(row["ti"]), -1)
        if col >= 0:
            scores[int(row["ni"]), col] = float(row["score"])
    return scores


def run():
    benchmark = load_benchmark_module()
    print(benchmark.section("MODEL ZOO — sklearn → deep"))
    print(f"  Inicio: {benchmark.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    data = benchmark.build_data()
    dates = data["dates"]
    feat = data["node_features"]
    splits = benchmark.make_splits(dates)

    print(f"\n  Grid: {feat.shape[0]} bairros | {len(dates)} dias | {feat.shape[2]} canais")
    print(f"  Splits: {len(splits)} folds mensais")

    models = [
        ("ST-GAT_Active", lambda d, tr, te: benchmark.run_stgat(d, te)),
        ("Naive_EWMA", benchmark.run_ewma),
        ("LightGBM_Rank", benchmark.run_lgbm_rank),
        ("LogReg", lambda d, tr, te: tabular_scores(d, tr, te, lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]), benchmark.HORIZON)),
        ("RandomForest", lambda d, tr, te: tabular_scores(d, tr, te, lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42
        ), benchmark.HORIZON)),
        ("ExtraTrees", lambda d, tr, te: tabular_scores(d, tr, te, lambda: ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42
        ), benchmark.HORIZON)),
        ("HistGB", lambda d, tr, te: tabular_scores(d, tr, te, lambda: HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
        ), benchmark.HORIZON)),
        ("LSTM", benchmark.run_lstm),
        ("TCN", benchmark.run_tcn),
    ]

    all_rows = []
    for train_range, test_range, fold_name in splits:
        test_start = dates[list(test_range)[0]].strftime("%d/%m/%Y")
        test_end = dates[list(test_range)[-1]].strftime("%d/%m/%Y")
        print(benchmark.section(f"Fold: {fold_name}  [{test_start} -> {test_end}]"))
        for model_name, model_fn in models:
            print(f"  [{model_name}]", end="  ", flush=True)
            start = time.time()
            try:
                scores = model_fn(data, train_range, test_range)
                result = benchmark.evaluate_model(scores, test_range, feat)
                p10 = result.get(10, 0.0)
                p20 = result.get(20, 0.0)
                status = "OK"
            except Exception as exc:
                p10, p20 = 0.0, 0.0
                status = f"ERRO: {exc}"
            elapsed = round(time.time() - start, 1)
            print(f"P@10={p10:.1f}%  P@20={p20:.1f}%  ({elapsed}s)  {status}")
            all_rows.append({"Modelo": model_name, "Fold": fold_name, "P@10": round(p10, 2), "P@20": round(p20, 2), "Tempo_s": elapsed})

    results = pd.DataFrame(all_rows)
    summary = (
        results.groupby("Modelo")[["P@10", "P@20", "Tempo_s"]]
        .mean()
        .round(2)
        .sort_values(["P@10", "P@20"], ascending=False)
        .reset_index()
    )

    summary_csv = os.path.join(OUT_DIR, "benchmark_model_zoo_summary.csv")
    results_csv = os.path.join(OUT_DIR, "benchmark_model_zoo_results.csv")
    report_txt = os.path.join(OUT_DIR, "benchmark_model_zoo_report.txt")

    results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(report_txt, "w", encoding="utf-8") as handle:
        handle.write("MODEL ZOO — sklearn → deep\n")
        handle.write(f"Data: {benchmark.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        handle.write("Resultados por fold:\n")
        handle.write(results.to_string(index=False))
        handle.write("\n\nResumo:\n")
        handle.write(summary.to_string(index=False))
        handle.write("\n")

    print(benchmark.section("RANKING FINAL"))
    print(summary.to_string(index=False))
    print(f"\n[OK] Relatorio: {report_txt}")


if __name__ == "__main__":
    run()
