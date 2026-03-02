# 計画: チーム編成による mock_lightgcn 実装

## コンテキスト

report.md の仕様定義が完了した。これをもとに、専門エージェントチームで並列実装を進める。

### 現在の状態（Phase 1 完了待ち）
- **mlops Task #2 完了**: mlflow 3.10.0 / dvc 3.66.1 / Python 3.13 / pandas 2.x で `uv sync` 成功
  - pyarrow は `>=23.0.1` + `[tool.uv] override-dependencies` で mlflow の `<22` 制約を上書き
  - pandas は `>=2.0,<3`（mlflow と互換）
- **researcher Task #1 進行中**: まとめ作業中。完了後にユーザーへ報告し、承認後に implementer を起動予定

## チーム構成

| エージェント名 | 役割 | 担当ファイル |
|---|---|---|
| **researcher** | Multi-Behavior LightGCN 論文調査 | `docs/research_notes.md` |
| **implementer** | コア実装 | `config.py`, `dataset.py`, `features.py`, `modeling/` |
| **mlops** | MLflow/DVC 実験管理基盤 | `pyproject.toml`, `modeling/train.py`（tracking部分）, `dvc.yaml`, `.mlflow/` |
| **tester** | テスト・CI/CD | `tests/`, `.github/workflows/` |
| **reviewer** | コードレビュー | 全ファイル（read-only レビュー→コメント） |

## 依存関係（実行順序）

```
Phase 1 (並列):
  researcher  →  docs/research_notes.md を作成
  mlops       →  mlflow/dvc のインストール設定（pyproject.toml 更新）

Phase 2 (researcher 完了後):
  implementer →  コア実装（research_notes.md を参照）

Phase 3 (implementer 完了後、並列):
  mlops       →  tracking コードを modeling/train.py に組み込み
  tester      →  テスト・CI/CD 作成

Phase 4 (全完了後):
  reviewer    →  コードレビュー & フィードバック
```

## 各エージェントのタスク詳細

### researcher のタスク
1. arxiv:2303.15720 の内容を WebFetch/WebSearch で調査
2. Wantedly ブログ (wantedly.com/.../1042160) の実装例を調査
3. 以下を `docs/research_notes.md` にまとめる：
   - Multi-Behavior LightGCN のアーキテクチャ詳細（行動別 GCN ブランチの接続方法）
   - 埋め込み統合方法（加重和 vs 連結）
   - マルチタスク損失の定式化
   - 実装上の注意点・工夫点

### implementer のタスク
`docs/research_notes.md` と `reports/report.md` を参照して実装。

1. **`src/mock_lightgcn/config.py`**
   - dataclass ベースの設定クラス
   - DataConfig（train_ratio=0.8, num_users=500, num_clients=200, etc.）
   - ModelConfig（embedding_dim=64, num_layers=3, lr=1e-3, batch_size=1024, epochs=200, patience=20, lambda_reg=1e-4）
   - EvalConfig（k_values=[5,10,20]）

2. **`src/mock_lightgcn/dataset.py`**
   - `generate_dummy_data(config: DataConfig) -> pd.DataFrame`
   - `split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]`
   - `encode_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]`

3. **`src/mock_lightgcn/features.py`**
   - `build_adj_matrix(df: pd.DataFrame, num_users: int, num_clients: int, behavior: str) -> torch.sparse.Tensor`
   - 対称正規化 D^{-1/2} A D^{-1/2} の実装

4. **`src/mock_lightgcn/modeling/__init__.py`** (空)

5. **`src/mock_lightgcn/modeling/train.py`**
   - `LightGCN(nn.Module)` クラス（ベースライン）
   - `MultiBehaviorLightGCN(nn.Module)` クラス（research_notes.md に基づく）
   - `bpr_loss(...)` 関数
   - `train(model, train_data, val_data, config)` 関数（Early stopping 込み）

6. **`src/mock_lightgcn/modeling/predict.py`**
   - `get_recommendations(model, user_ids, train_df, num_clients, k) -> dict`
   - 訓練データ既出 client の除外処理

7. **`src/mock_lightgcn/plots.py`**
   - 学習曲線の描画

### mlops のタスク
1. `pyproject.toml` に mlflow, dvc を追加（`uv add` 相当の編集）
2. `uv sync` を実行して lockfile を更新
3. DVC 初期化設定（`.dvc/config` でローカルストレージ設定）
4. `dvc.yaml` でパイプライン定義（generate_data → train → evaluate）
5. `modeling/train.py` に MLflow tracking を組み込む：
   - `mlflow.set_experiment("lightgcn")`
   - ハイパーパラメータ、エポックごとのメトリクス、モデルアーティファクトのログ

### tester のタスク
1. **`tests/test_dataset.py`**
   - `generate_dummy_data` のスモークテスト（shape, dtypes, 制約）
   - `split_train_test` の比率・コールドスタート除外テスト
   - `encode_ids` の 0-indexed 変換テスト

2. **`tests/test_features.py`**
   - `build_adj_matrix` の shape・対称性・正規化テスト

3. **`tests/test_modeling.py`**
   - `LightGCN` / `MultiBehaviorLightGCN` の forward pass スモークテスト（出力 shape）
   - `bpr_loss` の単体テスト

4. **`.github/workflows/ci.yml`**
   - push/PR トリガー
   - uv でインストール → ruff lint → pytest の実行

### reviewer のタスク
全実装完了後:
1. report.md の仕様と実装の整合確認
2. ruff lint/format に問題がないか確認
3. 型ヒントや docstring の漏れを確認
4. テストカバレッジの確認
5. フィードバックを `docs/review_notes.md` にまとめる

## 重要ファイルパス

- `reports/report.md` — 仕様書（全エージェントの参照元）
- `src/mock_lightgcn/__init__.py` — 現存するほぼ唯一のソースファイル
- `pyproject.toml` — 依存パッケージ管理（mlops が更新）
- `tests/test_data.py` — 現状 stub（tester が置き換える）

## Phase 1 完了後の引き継ぎ手順（context clear 前）

researcher の報告を確認・承認したあと、以下を実行してから `/clear` する。

### 1. グローバル設定をプロジェクトに移行
`~/.claude/settings.json` の内容をプロジェクトの `.claude/settings.json` にマージする。

**移行後の `.claude/settings.json`（プロジェクト）:**
```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "enabledPlugins": {
    "context7@claude-plugins-official": true,
    "document-skills@anthropic-agent-skills": true,
    "example-skills@anthropic-agent-skills": true
  }
}
```

**移行後の `~/.claude/settings.json`（グローバル）:**
```json
{
  "autoUpdatesChannel": "latest"
}
```
（`enabledPlugins` はグローバルから削除）
```

### 2. memory ファイルを更新
`C:\Users\aruto\.claude\projects\C--Users-aruto-dev-mock-lightgcn\memory\MEMORY.md` に現在の進行状況を記録しておく（context clear 後も引き継げるよう）。

### 3. `/clear` で context をリセット
新しいコンテキストで Phase 2 以降を再開する。

## 検証方法

```bash
make requirements   # uv sync が通ること
make lint           # ruff エラーなし
make test           # 全テストが通ること
dvc repro           # パイプラインが通ること（データ生成 → 学習 → 評価）
mlflow ui           # 実験ログが閲覧できること
```
