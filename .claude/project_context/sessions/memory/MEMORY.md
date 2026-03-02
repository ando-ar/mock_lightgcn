# mock_lightgcn Project Memory

## Project Overview
- **Goal**: LightGCN / Multi-Behavior LightGCN 推薦システム（人材業界: 求職者→企業推薦）
- **Stack**: Python 3.13, uv, PyTorch (CPU), pandas, MLflow, DVC
- **Package manager**: `uv` only (no pip/poetry)
- **Linter/formatter**: `ruff` (line-length=99, src=["src"])

## Current State (Phase 4 Complete - 2026-03-02)

### All phases DONE
- ✅ researcher: `docs/research_notes.md` created (MB-CGCN arXiv:2303.15720)
- ✅ mlops: mlflow/dvc in pyproject.toml, DVC initialized, dvc.yaml written
- ✅ implementer: All core files implemented
- ✅ tester: 26 tests, all passing
- ✅ reviewer: `docs/review_notes.md` created

### Key files
```
src/mock_lightgcn/
  config.py          # DataConfig, ModelConfig, EvalConfig dataclasses
  dataset.py         # generate_dummy_data, split_train_test, encode_ids
  features.py        # build_adj_matrix (D^-1/2 A D^-1/2 normalized sparse)
  plots.py           # plot_training_curve
  modeling/
    __init__.py      # empty
    train.py         # LightGCN, MultiBehaviorLightGCN (cascade), bpr_loss, train()
    predict.py       # get_recommendations()
tests/
  test_data.py       # config defaults
  test_dataset.py    # 9 tests for dataset functions
  test_features.py   # 8 tests for build_adj_matrix
  test_modeling.py   # 8 tests for models and bpr_loss
docs/
  research_notes.md  # MB-CGCN architecture, cascade, weighted sum
  review_notes.md    # code review with improvement proposals
dvc.yaml             # generate_data → train → evaluate pipeline
.github/workflows/ci.yml  # push/PR: uv install → ruff → pytest
```

## Architecture Details

### MultiBehaviorLightGCN (cascade structure from MB-CGCN)
- 4 behaviors: apply, pass, offer, offer_accept
- Cascade: behavior b output → W_b linear transform → behavior b+1 initial embedding
- `detach()` used to prevent gradient backflow through cascade
- `forward()` returns `(behavior_embs: dict, (user_emb, item_emb))`
- Weighted sum: apply=1, pass=2, offer=3, offer_accept=4 (normalized)

### Training
- Multi-task BPR: per-behavior embeddings for per-behavior losses
- Behavior loss weights: same 1:2:3:4
- Early stopping: evaluate every 10 epochs, patience=20 epochs (patience_counter += 10)
- MLflow tracking: hyperparams, epoch metrics, model artifact

### ruff config (pyproject.toml)
```toml
[tool.ruff]
line-length = 99
src = ["src"]
include = ["pyproject.toml", "src/**/*.py", "tests/**/*.py"]
```

## Known TODOs (from review_notes.md)
1. nDCG@k and mAP@k metrics not yet implemented (only Precision@k)
2. `compute_metrics` has O(N) per-user train masking (optimize with set)
3. matplotlib not explicitly in pyproject.toml dependencies
4. No separate val/test split (current code uses test_df as val_df in DVC pipeline)

## DVC Pipeline
```
generate_data → data/processed/train.parquet + test.parquet
train         → models/best_model.pt
evaluate      → reports/metrics.json
```

## MLflow
- Experiment: "lightgcn"
- Tracked: embedding_dim, lr, batch_size, epochs, patience, lambda_reg, model_type
- Per-epoch metrics: train_loss, val_precision@k
- Artifact: best_model.pt
