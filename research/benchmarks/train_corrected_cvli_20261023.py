"""Scratch ST-GAT v5 and Poisson walk-forward on corrected CVLI tensors only.
Uses the existing reviewed rolling-origin engine, but does NOT load or write
production data/models. All folds initialize the neural model from random
weights and lock out future target data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone
import torch
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import rolling_origin_cvli_retrain_20261022 as engine
SOURCE = ROOT / "data/runtime/research/cvli_rebuilt_20261022/all_official/processed_fortaleza.pkl"
OUT = HERE / "cvli_corrected_training_20261023"
CHECKPOINTS = OUT / "exp_models"
SOURCE_EXPECTED_SHA256 = "aa32f5fe829ab8345d68e67e2a19523f3886d8740c2f2b392943d29324807d24"
def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=(2024, 2025, 2026), required=True, type=int)
    args = parser.parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing corrected tensor: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    full = engine.load(SOURCE)
    assert len(full["nodes_gdf"]) == 121
    source_hash = digest(SOURCE)
    assert source_hash == SOURCE_EXPECTED_SHA256, "Corrected tensor changed; rebuild and version experiment"
    dates = full["dates"]
    print("CORRECTED_RETRAIN_BEGIN", "fold",args.fold,"tensor_sha256",source_hash,
          "nodes",len(full["nodes_gdf"]),"days",len(dates),flush=True)
    engine.OUT = OUT
    orig_train = engine.train_stgat
    orig_poisson = engine.poisson_fit
    def save_train(full, indices, tr, val, year, max_epochs=6):
        assert year == args.fold
        model,learning,best_epoch = orig_train(full,indices,tr,val,year,max_epochs=max_epochs)
        payload = dict(
            model="DeepSTGAT_v5", state_dict=model.state_dict(),
            in_channels=engine.CHANNELS, num_nodes=len(indices),
            time_steps=engine.WINDOW, seed=engine.SEED+year,
            fold=year, best_epoch=best_epoch,
            source_tensor_sha256=source_hash,
            train_end=f"{year-1}-12-31",
            validation_starts=[str(full["dates"][i]) for i in val],
            train_samples=len(tr),
            training_nodes=[str(full["nodes_gdf"].iloc[i]["name"]) for i in indices],
            weights_not_from_production=True,
        )
        dest=CHECKPOINTS / f"stgat_v5_scratch_cvli_corrected_fold_{year}.pth"
        torch.save(payload,dest)
        print("CHECKPOINT_SAVED",dest,"bytes",dest.stat().st_size,flush=True)
        return model,learning,best_epoch
    def save_poisson(full,tr,dates):
        payload=orig_poisson(full,tr,dates)
        dest=CHECKPOINTS / f"poisson_cvli_corrected_fold_{args.fold}.pkl"
        with dest.open("wb") as handle:
            pickle.dump({"source_tensor_sha256":source_hash, **payload},handle)
        print("POISSON_SAVED",dest,"bytes",dest.stat().st_size,flush=True)
        return payload
    engine.train_stgat=save_train
    engine.poisson_fit=save_poisson
    try:
        engine.simulate_fold(args.fold,full)
    finally:
        engine.train_stgat=orig_train
        engine.poisson_fit=orig_poisson
    p=OUT/f"fold_{args.fold}_metadata.json"
    metadata=json.loads(p.read_text(encoding="utf8"))
    metadata.update({
        "corrected_source_file":str(SOURCE.relative_to(ROOT)),
        "corrected_source_sha256":source_hash,
        "trained_from_scratch":True,
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "cvli_dedup_audit":"docs/AUDITORIA_CVLI_MYSQL_SSPDS_20261023.md",
        "previous_uncorrected_benchmarks_invalid_for_final_comparison":True,
    })
    p.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf8")
    print("CORRECTED_RETRAIN_SUCCESS",args.fold,flush=True)
if __name__ == "__main__":
    main()
