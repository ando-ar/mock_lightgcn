# エージェント共通ガイド

このファイルは、リポジトリ上でコードを操作する際に、Claude / Codex の両方へ共有するガイダンスです。

## コマンド

```bash
# 環境構築
make create_environment   # Python 3.14 + uv の仮想環境を作成
make requirements         # uv sync で依存パッケージをインストール

# コード品質
make lint                 # ruff format --check と ruff check
make format               # ruff check --fix と ruff format

# テスト
make test                 # tests/ に対して pytest を実行
pytest tests/test_data.py # 単一テストファイルを実行

# クリーンアップ
make clean                # __pycache__ と生成ファイルを削除
```

## アーキテクチャ

このプロジェクトは **LightGCN** (Light Graph Convolutional Network) を用いた推薦システムです。
Cookiecutter Data Science テンプレートに沿って構成されています。

- パッケージ管理: `uv`（`pip` / `poetry` は使用しない）
- リンター/フォーマッター: `ruff`（行長 99、isort 有効、first-party: `mock_lightgcn`）
- ビルドバックエンド: `flit_core`

## パッケージ構成 (`src/mock_lightgcn/`)

- `config.py`: 設定と定数
- `dataset.py`: データの取得・読み込み
- `features.py`: 特徴量エンジニアリング
- `modeling/train.py`: 学習
- `modeling/predict.py`: 推論
- `plots.py`: 可視化

## データディレクトリ

- `data/raw/`: 元データ（変更しない）
- `data/interim/`: 中間データ
- `data/processed/`: モデリング用の最終データ
- `data/external/`: 外部データ

学習済みモデルは `models/`、図表は `reports/figures/` に保存します。

## ルール

- 既存の処理フローを優先し、機能追加は最小差分で行う
- 変更時は `make lint` と `make test` を実行する
- 生データは直接編集しない
- 機密情報をコミットしない（`.env` と認証情報を除外）

## Issue駆動 開発フロー

このリポジトリでは、作業は原則 GitHub Issue を起点に進める。

### 基本方針

- 1 Issue = 1 目的（小さく分割し、受け入れ条件を明確化する）
- 実装前に Issue を作成し、完了定義（DoD）を先に決める
- 変更は必ず Issue 番号と紐づける（branch / commit / PR）
- 仕様変更や追加要件は Issue コメントで合意を残す
- 直接 `main` に push しない（PR 経由）

### Issue 作成ルール

- タイトル形式: `<type>(<scope>): <summary>`
- type は `feat` `fix` `refactor` `chore` `docs` `test` を使用
- 本文に以下を必ず含める
- 背景
- 対応内容
- 受け入れ条件
- 影響範囲（任意）
- 優先度（P1/P2/P3）

### ブランチ運用

- ブランチ名: `issue-<番号>-<short-topic>`
- 例: `issue-4-metrics-map-ndcg`
- 1 ブランチで複数 Issue を同時に扱わない

### 実装時の必須手順

- Issue の受け入れ条件を満たす最小差分で実装する
- 実装後に以下を実行する
- `make lint`
- `make test`
- 必要なら `dvc repro`
- 結果を PR または Issue コメントに記録する

### コミット規約

- 形式: `<type>: <summary> (#<issue番号>)`
- 例: `feat: add mAP@k and nDCG@k metrics (#4)`
- 大きな変更は論理単位でコミットを分ける

### PR ルール

- PR タイトルに Issue 番号を含める
- PR 本文に以下を記載する
- 目的（Issue へのリンク）
- 変更内容
- 検証結果（lint/test/dvc）
- 残課題（あれば）
- マージ条件
- CI 緑
- 受け入れ条件をすべて満たす
- レビュー指摘への対応完了

### 完了条件（Definition of Done）

- Issue の受け入れ条件を満たしている
- `make lint` / `make test` が成功
- 必要なテストが追加・更新されている
- ドキュメント更新が必要な場合は反映済み
- Issue/PR に検証ログが残っている

### 優先順位の基本

- P1: 学習や評価の正しさに直結する不具合・欠落
- P2: 性能改善・開発体験改善
- P3: 整理・軽微な改善

### エージェント運用メモ

- 作業開始時に「どの Issue を解くか」を明示する
- 作業完了時に「何を変更し、どう検証したか」を簡潔に報告する
- 未解決事項は次の Issue として分離する
