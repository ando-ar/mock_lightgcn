# Multi-Behavior LightGCN 研究ノート

## 参考文献

- **MB-CGCN 論文**: Wei et al., "Multi-Behavior Recommendation with Cascading Graph Convolution Networks", WWW 2023. [arXiv:2303.15720](https://arxiv.org/abs/2303.15720)
- **Wantedly 実装ブログ**: [マルチビヘイビア推薦の実装](https://www.wantedly.com/companies/wantedly/post_articles/1042160)
- **LightGCN 原論文**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020. [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)

---

## 1. Multi-Behavior LightGCN アーキテクチャ詳細

### 1.1 基盤: LightGCN の伝播式

LightGCN は GCN から特徴変換と非線形活性化を除去し、近傍集約のみを行う軽量モデルである。

**伝播式 (各レイヤー l):**

```
e_u^(l+1) = Σ_{i ∈ N_u} (1 / sqrt(|N_u| * |N_i|)) * e_i^(l)
e_i^(l+1) = Σ_{u ∈ N_i} (1 / sqrt(|N_i| * |N_u|)) * e_u^(l)
```

行列形式では:

```
E^(l+1) = D^{-1/2} A D^{-1/2} E^(l)
```

ここで:
- `A` はユーザー-アイテム二部グラフの隣接行列 (対称化済み)
- `D` は次数行列 (対角行列)
- `E^(l)` は第 l レイヤーの埋め込み行列

### 1.2 レイヤー埋め込みの統合

**原論文 LightGCN**: 全レイヤーの均等平均

```
e_u = (1 / (L+1)) * Σ_{l=0}^{L} e_u^(l)
```

**MB-CGCN**: 全レイヤーの単純な和 (sum)

```
e_u^(b) = Σ_{l=0}^{L} e_u^(b,l)
e_i^(b) = Σ_{l=0}^{L} e_i^(b,l)
```

MB-CGCN では `1/(L+1)` の係数を省略している。予測スコアには影響しない (内積のスケーリングのみ) が、本プロジェクトでは原論文に従い `1/(L+1)` の平均を推奨する。

### 1.3 行動別 GCN ブランチの構成

MB-CGCN では、各行動 (behavior) ごとに独立した LightGCN ブロックを持つ。本プロジェクトでは以下の 4 行動を対象とする:

1. **apply** (応募)
2. **pass** (書類通過)
3. **offer** (オファー)
4. **offer_accept** (オファー受諾)

各行動 b に対して:
- 独立した隣接行列 `A_b` を構築 (当該行動が発生した user-client ペアのみ)
- 独立した LightGCN で L 層の伝播を実行
- 行動ごとのユーザー・アイテム埋め込み `e_u^(b)`, `e_i^(b)` を出力

### 1.4 カスケード構造 (MB-CGCN の特徴)

MB-CGCN の核心は、行動間の依存関係をカスケード (連鎖) として活用する点にある。

行動チェーン: apply -> pass -> offer -> offer_accept

前段の行動 b で学習した埋め込みを、次の行動 b+1 の初期埋め込みとして使用する:

```
e_u^(b+1, 0) = W_u^b * e_u^(b)
e_i^(b+1, 0) = W_i^b * e_i^(b)
```

ここで `W_u^b`, `W_i^b` は学習可能な線形変換行列 (bias なし) である。

**重要な実装ポイント**: Wantedly ブログでは、変換後の埋め込みを `detach()` して勾配の逆流を防ぐことが推奨されている。

---

## 2. 埋め込み統合方法

### 2.1 MB-CGCN 原論文: 全行動の単純和

```
e_u = Σ_{b=1}^{B} e_u^(b)
e_i = Σ_{b=1}^{B} e_i^(b)
```

予測スコア:
```
y_hat_{ui} = e_u^T * e_i
```

### 2.2 本プロジェクトでの推奨: 加重和

report.md の仕様に基づき、行動の重要度に応じた加重和を採用する:

```
e_u = Σ_{b=1}^{B} w_b * e_u^(b)
e_i = Σ_{b=1}^{B} w_b * e_i^(b)
```

重み:
- apply: w = 1
- pass: w = 2
- offer: w = 3
- offer_accept: w = 4

### 2.3 統合方法の比較

| 方法 | 利点 | 欠点 |
|------|------|------|
| **単純和** | シンプル、パラメータ不要 | 行動の重要度差を反映できない |
| **加重和** (推奨) | 行動の重要度を反映、チューニング可能 | 重みの設定にドメイン知識が必要 |
| **連結 (concat)** | 行動間の独立性を保持 | 次元が B 倍に増大、下流に線形変換が必要 |
| **Attention** | データ駆動で重みを学習 | パラメータ増、小データで不安定 |

本プロジェクトでは加重和を第一選択とする。

---

## 3. マルチタスク損失の定式化

### 3.1 各行動の BPR ロス

行動 b に対する BPR ロス:

```
L_b = Σ_{(u,i,j) ∈ O_b} -ln σ(y_{ui}^(b) - y_{uj}^(b))
```

ここで:
- `O_b = {(u, i, j) | user u が行動 b で item i とインタラクションあり、item j はなし}`
- `y_{ui}^(b) = (e_u^(b))^T * e_i^(b)` (行動 b の埋め込みによる予測スコア)
- `σ` はシグモイド関数

### 3.2 ネガティブサンプリング

report.md の仕様に基づく:
- 各ユーザーに対して、**レコメンド済みかつ対象行動を取っていない client** をネガティブとしてサンプリング
- 均一ランダムサンプリング

### 3.3 マルチタスク損失

**MB-CGCN 原論文**: ターゲット行動 (最終行動) のみで損失を計算。マルチタスク学習は行わない。

**本プロジェクトの方針 (report.md)**: 全行動の BPR ロスを重み付きで合算するマルチタスク学習を採用:

```
L_total = Σ_{b=1}^{B} α_b * L_b + λ * ||Θ||^2
```

損失重み α_b:
- apply: α = 1
- pass: α = 2
- offer: α = 3
- offer_accept: α = 4

L2 正則化:
- λ = 1e-4
- Θ は初期埋め込み (e_u^(0), e_i^(0)) と変換行列 W を含む全学習パラメータ

### 3.4 損失重みの根拠

- 上位行動 (offer_accept) ほどビジネス価値が高い
- 上位行動ほどデータが疎 (正例が少ない) ため、損失を大きく重み付けしてバランスを取る
- apply:pass:offer:offer_accept = 1:2:3:4 は、行動チェーンの段階に比例した線形スケーリング

---

## 4. 実装上の注意点・工夫

### 4.1 隣接行列の対称正規化

二部グラフの隣接行列を対称正規化する手順:

```python
# 1. user-item 二部グラフの隣接行列を構築
#    A の形状: (num_users + num_clients) x (num_users + num_clients)
#    上三角: user -> client のエッジ
#    下三角: client -> user のエッジ (対称化)

# 2. 対称正規化 D^{-1/2} A D^{-1/2}
#    D は次数行列 (対角要素 = 各ノードの次数)
```

実装の疑似コード:

```python
import torch
from torch_sparse import SparseTensor

def build_norm_adj(user_ids, item_ids, num_users, num_items):
    """対称正規化された隣接行列を構築する"""
    # user-item エッジ (user offset: 0, item offset: num_users)
    row = torch.cat([user_ids, item_ids + num_users])
    col = torch.cat([item_ids + num_users, user_ids])

    num_nodes = num_users + num_items
    adj = SparseTensor(row=row, col=col,
                       sparse_sizes=(num_nodes, num_nodes))

    # 次数を計算
    deg = adj.sum(dim=1)  # shape: (num_nodes,)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # D^{-1/2} A D^{-1/2}
    adj = adj * deg_inv_sqrt.view(-1, 1)  # 行方向
    adj = adj * deg_inv_sqrt.view(1, -1)  # 列方向

    return adj
```

### 4.2 疎行列の効率的な扱い方

- **PyTorch の `torch.sparse`** または **`torch_sparse`** パッケージを使用
- 隣接行列は COO 形式 (`torch.sparse_coo_tensor`) で構築し、`torch.sparse.mm` で行列積を計算
- メモリ効率のために、隣接行列は事前に正規化して保持 (エポックごとに再計算しない)

```python
# 疎行列での伝播 (効率的)
def propagate(adj_norm, embeddings):
    """LightGCN の 1 レイヤー伝播"""
    return torch.sparse.mm(adj_norm, embeddings)
```

### 4.3 LightGCN の完全な伝播

```python
def lightgcn_forward(adj_norm, init_embeddings, num_layers):
    """LightGCN の順伝播: 全レイヤーの平均"""
    all_embeddings = [init_embeddings]
    current = init_embeddings

    for _ in range(num_layers):
        current = torch.sparse.mm(adj_norm, current)
        all_embeddings.append(current)

    # レイヤー平均
    final = torch.stack(all_embeddings, dim=0).mean(dim=0)
    return final
```

### 4.4 Multi-Behavior LightGCN の全体フロー

```python
class MultiBehaviorLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, behaviors):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.behaviors = behaviors  # ['apply', 'pass', 'offer', 'offer_accept']

        # 共有初期埋め込み
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # カスケード変換行列 (行動数 - 1 個)
        self.transforms = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(len(behaviors) - 1)
        ])
        for t in self.transforms:
            nn.init.xavier_uniform_(t.weight)

    def forward(self, adj_matrices):
        """
        adj_matrices: dict[behavior_name -> normalized sparse adj matrix]
        """
        all_behavior_embeddings = []

        # 最初の行動の初期埋め込み
        init_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        current_init = init_emb

        for b_idx, behavior in enumerate(self.behaviors):
            # 行動 b の LightGCN 伝播
            emb_b = lightgcn_forward(
                adj_matrices[behavior], current_init, self.num_layers
            )
            all_behavior_embeddings.append(emb_b)

            # 次の行動へカスケード (最後の行動以外)
            if b_idx < len(self.behaviors) - 1:
                current_init = self.transforms[b_idx](emb_b.detach())

        return all_behavior_embeddings

    def predict(self, all_behavior_embeddings, user_ids, item_ids, weights):
        """加重和で統合して予測"""
        user_emb = sum(
            w * emb[user_ids]
            for w, emb in zip(weights, all_behavior_embeddings)
        )
        item_emb = sum(
            w * emb[self.num_users + item_ids]
            for w, emb in zip(weights, all_behavior_embeddings)
        )
        return (user_emb * item_emb).sum(dim=-1)
```

### 4.5 BPR ロスの実装

```python
def bpr_loss(pos_scores, neg_scores):
    """BPR ロス: 正例スコアが負例スコアより高くなるように学習"""
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

def multi_task_loss(model, all_behavior_embeddings, behavior_samples,
                    loss_weights, reg_lambda):
    """マルチタスク BPR ロス"""
    total_loss = 0
    for b_idx, (behavior, samples) in enumerate(behavior_samples.items()):
        user_ids, pos_ids, neg_ids = samples
        emb = all_behavior_embeddings[b_idx]

        user_emb = emb[user_ids]
        pos_emb = emb[model.num_users + pos_ids]
        neg_emb = emb[model.num_users + neg_ids]

        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)

        total_loss += loss_weights[behavior] * bpr_loss(pos_scores, neg_scores)

    # L2 正則化
    reg_loss = reg_lambda * (
        model.user_embedding.weight.norm(2).pow(2) +
        model.item_embedding.weight.norm(2).pow(2) +
        sum(t.weight.norm(2).pow(2) for t in model.transforms)
    )

    return total_loss + reg_loss
```

### 4.6 Early Stopping の実装上の注意

```python
class EarlyStopping:
    def __init__(self, patience=20, eval_interval=10):
        self.patience = patience
        self.eval_interval = eval_interval
        self.best_metric = -float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_state = None

    def step(self, epoch, metric, model):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = deepcopy(model.state_dict())
        else:
            self.counter += self.eval_interval  # eval_interval 刻みでカウント

        return self.counter >= self.patience
```

注意点:
- `eval_interval=10` エポックごとに評価するため、`counter` は 10 刻みで増加
- `patience=20` の場合、実質 2 回連続で改善しなければ停止
- 最良エポックのモデル状態を保存し、学習終了後にロードする

---

## 5. 本プロジェクトにおける設計判断のまとめ

| 項目 | MB-CGCN 原論文 | 本プロジェクトの方針 |
|------|----------------|---------------------|
| 行動数 | 可変 | 4 (apply, pass, offer, offer_accept) |
| レイヤー統合 | sum | mean (1/(L+1)) |
| 行動間の接続 | カスケード + 変換行列 | カスケード + 変換行列 |
| 埋め込み統合 | 全行動の sum | 加重和 (1:2:3:4) |
| 損失関数 | ターゲット行動のみ BPR | マルチタスク BPR (重み 1:2:3:4) |
| 正則化 | L2 | L2 (λ=1e-4) |
| Optimizer | - | Adam (lr=1e-3) |
| 埋め込み次元 | - | 64 |
| 伝播レイヤー数 | - | 3 |
| バッチサイズ | - | 1024 |
| Early stopping | - | patience=20, eval_interval=10 |

---

## 6. ベースライン LightGCN との差分

Multi-Behavior LightGCN はベースライン LightGCN に対して以下の拡張を加える:

1. **隣接行列の複数化**: 単一の隣接行列 -> 行動ごとに独立した 4 つの隣接行列
2. **カスケード構造**: 行動チェーンに沿った特徴変換 (W 行列)
3. **埋め込みの加重統合**: 行動ごとの埋め込みを重み付きで集約
4. **マルチタスク損失**: 全行動の BPR ロスを重み付きで合算

ベースライン LightGCN は apply 行動のみ (または全行動を統合した単一グラフ) で学習し、Multi-Behavior 版との性能比較を行う。
