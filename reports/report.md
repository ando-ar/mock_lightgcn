# report

## introduction
人材業界における求職者に対する推薦システムを対象に、LightGCNやその派生形のモデルの性能を知りたい。

## method

### dataset

データセットでは、次のカラム構成を想定する。
- user_id (求職者ID)
- client_id (企業ID)
- recommend_date (求職者へのレコメンド日。必ず存在する)
- apply_date (求職者が応募した日。応募していれば存在する)
- pass_date (企業が書類通過させた日。書類通過していれば存在する)
- offer_date(企業が求職者にオファーを出した日。オファーがあれば存在する)
- offer_accept_date (求職者が企業のオファーを受諾した日。オファーの受諾があれば存在する)

user_idとclient_idの組み合わせで一行のレコードであり、重複は存在しない。
user_idごとに、レコメンドされたclient_idの数はばらつきが生じる(ただし1件以上は必ず存在する)。
user_idに対してoffer_accept_dateは一つ以下。
recommend_dateに従ってこれを訓練データとテストデータに分割する。

動作確認用にダミーデータも作成できるようにする。
P(応募|レコメンド)、P(書類通過|応募)、P(オファー|書類通過)、P(オファー受諾|書類通過)はパラメータで指定できるようにする。

#### 実装仕様

**訓練/テスト分割**
- recommend_date の昇順にソートし、先頭 80% を訓練データ、残り 20% をテストデータとする（デフォルト比率）
- 分割比率はパラメータ `train_ratio`（デフォルト: 0.8）で変更可能
- テストデータに含まれる user_id が訓練データに存在しない場合（コールドスタートユーザー）、そのレコードは評価から除外する

**ID エンコーディング**
- 訓練データに登場する user_id と client_id をそれぞれ 0-indexed の整数にマッピングする
- テストデータに訓練データ未登場の ID が現れた場合、そのレコードは除外する（未知IDへの推論は行わない）

**ダミーデータ生成パラメータのデフォルト値**
- `num_users`: 500
- `num_clients`: 200
- `num_recommendations_per_user`: Poisson(λ=20) でサンプリング（最小1件）
- `p_apply`: P(応募|レコメンド) = 0.3
- `p_pass`: P(書類通過|応募) = 0.5
- `p_offer`: P(オファー|書類通過) = 0.4
- `p_offer_accept`: P(オファー受諾|オファー) = 0.6
- `recommend_date` の生成範囲: 2024-01-01 〜 2024-12-31 の一様分布

**インタラクション行列の構築**
- apply, pass, offer, offer_accept の 4 行動それぞれについて独立した隣接行列 A を構築する
- 各行列は user×client の二部グラフとして表現し、対称正規化 D^{-1/2} A D^{-1/2} を適用する

### model

- LightGCN(ベースライン)
- Mulit-Behavior LightGCN

Mulit-Behavior LightGCNに関する論文と実装例は以下を参考にする。
https://arxiv.org/abs/2303.15720
https://www.wantedly.com/companies/wantedly/post_articles/1042160

そのほかについては、実験の進行に伴って追加する。

#### 実装仕様

**LightGCN ハイパーパラメータ**
- `embedding_dim`: 64
- `num_layers`: 3
- 学習率: 1e-3（Adam optimizer）
- バッチサイズ: 1024
- エポック数: 200
- Early stopping: バリデーション指標が 20 epoch 連続で改善しない場合に学習を打ち切る
- 正則化係数 λ: 1e-4（L2 正則化）

**損失関数**
- BPR (Bayesian Personalized Ranking) ロスを使用する
- ネガティブサンプリング: 均一ランダム。各ユーザーに対してレコメンド済みかつ対象行動を取っていない client をネガティブとしてサンプリングする

**Multi-Behavior LightGCN の構成**（arxiv:2303.15720 に基づく）
- apply, pass, offer, offer_accept の各行動について独立した GCN ブランチを持ち、行動ごとの埋め込みを生成する
- 各行動の埋め込みを加重和で統合し、最終的なユーザー・アイテム埋め込みを得る
- マルチタスク学習として各行動に対する BPR ロスを合算する。上位行動（offer_accept）の損失重みを大きく設定する（例: offer_accept=4, offer=3, pass=2, apply=1）

### metrics

- 応募に対するPrecision@k, mAP@k
- 書類通過に対するPrecision@k, mAP@k
- オファーに対するPrecision@k, mAP@k
- オファー受諾に対するPrecision@k, mAP@k
- オファー受諾>オファー>書類通過>応募>非応募 としたときのnDCG@k

#### 実装仕様

**k の値**
- k = 5, 10, 20 を標準とする

**Precision@k の定義**
- 上位 k 件の推薦リスト中に正例（対象行動を取った client）が何件含まれるかの割合
- Precision@k = (上位 k 件中の正例数) / k

**mAP@k の定義**
- ユーザーごとに Average Precision@k を算出し、その平均を取る
- Average Precision@k = Σ_{i=1}^{k} [i番目が正例] × Precision@i / min(正例総数, k)
- 正例が 0 件のユーザーは平均計算から除外する

**nDCG@k のゲイン関数**
- 行動の関連度スコア: offer_accept=4, offer=3, pass=2, apply=1, 非応募=0
- gain(i) = (2^{relevance_i} − 1) / log_2(i + 1)（i は 1-indexed）
- Ideal DCG は関連度スコアの降順に並べたときの DCG で正規化する

**ランキング候補の定義**
- 訓練データに含まれる（レコメンド済みの）client は推薦候補から除外する
- テストデータの全 client を候補とする方式を第一選択とする

**評価タイミング**
- 10 epoch ごとにバリデーションセットで評価し、Early stopping の判定に使用する
- 最終エポック（または Early stopping 後の最良エポック）でテストセットを評価する

## result

### データのサマリ
- 訓練データ、テストデータの行数
- 訓練データ、テストデータにおけるuser_idとclient_idのユニーク数
- 訓練データ、テストデータにおける応募総数、書類通過総数、オファー総数、オファー受諾総数

### 各手法の比較

## discussion

## conclusion