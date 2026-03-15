# クイックスタート

このガイドでは、リポジトリのクローンから学習実行までの手順を説明します。

## 前提条件

- Python 3.13
- [uv](https://docs.astral.sh/uv/) がインストール済みであること

```bash
# uv が未インストールの場合
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. リポジトリのクローン

```bash
git clone https://github.com/ando-ar/mock_lightgcn.git
cd mock_lightgcn
```

## 2. 仮想環境の作成

```bash
make create_environment
source .venv/bin/activate
```

## 3. 依存パッケージのインストール

```bash
make requirements
```

## 4. コード品質チェック

```bash
# フォーマット・lint チェック
make lint

# 型チェック
make typecheck
```

## 5. テストの実行

```bash
make test
```

すべてのテストが `PASSED` になれば環境構築は完了です。

## 6. 学習の実行

```bash
uv run python -m mock_lightgcn.modeling.train
```

学習済みモデルは `models/` ディレクトリに保存されます。
学習ログは `mlruns/` に記録されます。

## 7. MLflow UI で結果を確認

学習後、以下のコマンドで MLflow UI を起動できます。

```bash
uv run mlflow ui
```

ブラウザで <http://localhost:5000> を開くと、`lightgcn` エクスペリメント配下に実行ログが表示されます。

確認できる主な情報:

| 項目 | 説明 |
|------|------|
| `train_loss` | エポックごとの学習損失 |
| `val_precision@10` | バリデーション Precision@10 |
| `val_ndcg@10` | バリデーション nDCG@10 |
| `val_map@10` | バリデーション mAP@10 |
| `best_val_precision@10` | 最良エポックの Precision@10 |
| モデルアーティファクト | `models/lightgcn_best.pt` |

ポートを変更したい場合:

```bash
uv run mlflow ui --port 5001
```

## 8. 次のステップ

- アーキテクチャの詳細は [`CLAUDE.md`](CLAUDE.md) を参照
- 設定パラメータの変更は [`src/mock_lightgcn/config.py`](src/mock_lightgcn/config.py) を編集
- DVC パイプラインの実行: `dvc repro`
- Issue 駆動の開発フローは [`CLAUDE.md`](CLAUDE.md) の「Issue駆動 開発フロー」を参照
