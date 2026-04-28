# Project Structure

## Root Directory
- `app.py`: Main application entry point (Flask).
- `GEMINI.md`: Current project status and rules.
- `TRAINING_LOG.md`: Detailed history of all training attempts.
- `README.md` / `GETTING_STARTED.md`: Documentation.

## Key Directories
- `src/core/`:
  - `architectures.py`: ST-GAT model definitions (DeepSTGAT_64, DeepSTGAT_80).
  - `orchestrator.py`: Logic for running inferences and managing models.
  - `champion_challenger.py`: Hybrid blend logic (EMA weight adjustment).
  - `training_vault.py`: MemPalace memory system.
- `scripts/training/Active/`:
  - `train_all_specialists.py`: Official training script for regional models.
- `tests/Sentinela/`:
  - `ROADMAP.md`: Sentinela development plan.
  - `train_validate_v3.py`: Validation script for LGBM.
  - `finetune_realtime_v1.py`: Real-time adjustment logic.
- `models/active/`:
  - Production-ready model files (`.pth` and `.pkl`).
- `data/`:
  - `processed/`: Serialized features for training.
  - `raw/`: Raw CSV data.
- `logs/`:
  - Training and system logs.
