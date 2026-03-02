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
