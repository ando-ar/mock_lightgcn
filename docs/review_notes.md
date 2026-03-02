# コードレビューノート

## レビュー日: 2026-03-02

---

## 1. report.md 仕様と実装の整合確認

### ✅ データセット (dataset.py)
- [x] user_id, client_id, recommend_date, apply_date, pass_date, offer_date, offer_accept_date の 7 カラム
- [x] (user_id, client_id) の組み合わせは一意（重複なし）
- [x] user_id ごとに offer_accept は最大 1 件
- [x] 行動依存関係: apply → pass → offer → offer_accept
- [x] recommend_date 昇順での 80/20 分割
- [x] コールドスタートユーザーの除外
- [x] 0-indexed エンコーディング

### ✅ 特徴量 (features.py)
- [x] 4 行動それぞれの独立した隣接行列
- [x] user×client 二部グラフとして表現
- [x] 対称正規化 D^{-1/2} A D^{-1/2}
- [x] torch.sparse.FloatTensor として返却

### ✅ モデル (modeling/train.py)

#### LightGCN (ベースライン)
- [x] embedding_dim=64, num_layers=3 (ModelConfig デフォルト)
- [x] Xavier 初期化
- [x] 全レイヤー埋め込みの平均（LightGCN 原論文通り）
- [x] 内積による推薦スコア

#### MultiBehaviorLightGCN
- [x] 4 行動に対する独立 GCN ブランチ
- [x] **カスケード構造**: 行動 b の出力を W_b で変換し行動 b+1 の初期埋め込みに利用
  - `detach()` を使用して勾配の逆流を防止 (Wantedly ブログ推奨通り)
- [x] 加重和による統合: apply:pass:offer:offer_accept = 1:2:3:4
- [x] forward が `(behavior_embs, (user_emb, item_emb))` を返す設計

#### BPR 損失
- [x] `-log(σ(pos_score - neg_score))`
- [x] L2 正則化（初期埋め込みのみ）
- [x] λ=1e-4 (ModelConfig デフォルト)

#### マルチタスク学習
- [x] 各行動の BPR ロスを行動ごとの per-behavior 埋め込みで計算
- [x] 損失重み: apply=1, pass=2, offer=3, offer_accept=4
- [x] 1 epoch 1 forward pass で全行動を同時最適化（効率的）

#### Early Stopping
- [x] 10 epoch ごとに apply 行動の Precision@10 で評価
- [x] patience=20 epochs（`patience_counter += 10` で管理）
- [x] 最良モデルを `models/best_model.pt` に保存

### ✅ MLflow Tracking (modeling/train.py)
- [x] `mlflow.set_experiment("lightgcn")`
- [x] ハイパーパラメータのログ（embedding_dim, lr, etc.）
- [x] エポックごとの train_loss ログ
- [x] 10 epoch ごとの val precision@k ログ
- [x] 最良モデルアーティファクトのログ

### ✅ 推論 (modeling/predict.py)
- [x] 訓練済み client の除外
- [x] top-k 推薦リストを dict で返却

### ✅ DVC パイプライン (dvc.yaml)
- [x] generate_data → train → evaluate の 3 ステージ
- [x] deps/outs の依存関係定義

### ✅ CI/CD (.github/workflows/ci.yml)
- [x] push/PR トリガー
- [x] uv セットアップ
- [x] ruff format check + ruff check
- [x] pytest 実行

---

## 2. ruff lint/format 確認

```
ruff format --check src/ tests/  → All checks passed
ruff check src/ tests/           → All checks passed
```

全ファイルで ruff エラーなし。

---

## 3. 型ヒントと docstring の確認

### 型ヒント
- [x] 全パブリック関数に型ヒントあり
- [x] `from __future__ import annotations` を適切に使用

### Docstring
- [x] 主要クラス・関数にはドキュメントあり
- `_build_behavior_pairs`, `_sample_negatives` など内部ヘルパーは省略可

---

## 4. テストカバレッジ確認

| テストファイル | テスト数 | カバー対象 |
|---|---|---|
| test_data.py | 1 | config デフォルト値 |
| test_dataset.py | 9 | generate_dummy_data, split_train_test, encode_ids |
| test_features.py | 8 | build_adj_matrix（shape, 対称性, 正規化, sparse） |
| test_modeling.py | 8 | LightGCN/MB-LightGCN forward, bpr_loss |
| **合計** | **26** | |

全 26 テスト PASSED。

---

## 5. 改善提案・TODO

### 優先度: 中

1. **compute_metrics の O(N) ボトルネック**
   ```python
   # 現状: 全 item_idx をループして train 除外
   for item_idx in range(num_items):
       if (user_idx, item_idx) in train_pairs_set:
   ```
   → ユーザーごとに `train_items_per_user` のセットを持ち、`scores[list(exclude)]=-inf` にする方が高速。

2. **nDCG, mAP メトリクスの未実装**
   - report.md では Precision@k に加え mAP@k と nDCG@k も指定されているが、
     現実装では Precision@k のみ実装。
   - Early stopping の判断基準として Precision@10 は適切だが、最終評価には nDCG も追加推奨。

3. **val_df / test_df の区別**
   - 現在の `train()` は `val_df` 引数だが DVC パイプラインでは test.parquet を渡している。
   - 将来的には val/test を明示的に分けることを推奨（例: val=70-80%, test=80-100%）。

### 優先度: 低

4. **plots.py の matplotlib import**
   - `matplotlib` は `pyproject.toml` の明示的依存に含まれていない（transitive として利用）。
   - 追加推奨: `"matplotlib>=3.0"` を pyproject.toml に明示。

5. **`offer_accept_date` 単一ユーザー制約の完全性**
   - 現状の `generate_dummy_data` では、1 ユーザーが複数の offer_accept を持つケースを
     `has_accepted` フラグで防いでいるが、`recommend_date` 順に生成しているわけではない。
   - 実データに合わせる場合は推薦日順にソートしてから適用が望ましい。

---

## 6. 総評

実装は report.md の仕様を概ね忠実に反映している。MB-CGCN の核心であるカスケード構造
（前行動の埋め込みを変換して次行動の初期値として利用）が正しく実装され、マルチタスク
BPR 損失も per-behavior 埋め込みを利用して計算されている。テストカバレッジも主要パスを
カバーしており、CI/CD も整備されている。

上記の「改善提案」のうち nDCG/mAP の実装は次のフェーズで対応することを推奨する。
