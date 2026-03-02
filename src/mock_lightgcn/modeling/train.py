from __future__ import annotations

import os

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from mock_lightgcn.config import EvalConfig, ModelConfig

BEHAVIOR_WEIGHTS = {"apply": 1, "pass": 2, "offer": 3, "offer_accept": 4}
BEHAVIORS = list(BEHAVIOR_WEIGHTS.keys())
BEHAVIOR_DATE_MAP = {
    "apply": "apply_date",
    "pass": "pass_date",
    "offer": "offer_date",
    "offer_accept": "offer_accept_date",
}


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(num_users + num_items, config.embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adj_matrix: torch.sparse.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = self.embedding.weight
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb = final_emb[: self.num_users]
        item_emb = final_emb[self.num_users :]
        return user_emb, item_emb

    def get_embeddings(
        self, adj_matrix: torch.sparse.FloatTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(adj_matrix)


class MultiBehaviorLightGCN(nn.Module):
    """Multi-Behavior LightGCN with cascade structure (MB-CGCN, arXiv:2303.15720).

    Each behavior has its own LightGCN branch. Behaviors are chained: the output
    of behavior b is linearly transformed and used as the initial embedding for
    behavior b+1. Final embeddings are a weighted sum of all behavior embeddings.
    """

    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = config.num_layers
        embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        # Cascade linear transforms: one per behavior transition (len(BEHAVIORS)-1)
        self.transforms = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(len(BEHAVIORS) - 1)
            ]
        )
        for t in self.transforms:
            nn.init.xavier_uniform_(t.weight)

    def _propagate(
        self, adj_matrix: torch.sparse.FloatTensor, init_emb: torch.Tensor
    ) -> torch.Tensor:
        """LightGCN propagation from given initial embeddings."""
        embs = [init_emb]
        current = init_emb
        for _ in range(self.num_layers):
            current = torch.sparse.mm(adj_matrix, current)
            embs.append(current)
        return torch.stack(embs, dim=1).mean(dim=1)

    def forward(
        self,
        adj_matrices: dict[str, torch.sparse.FloatTensor],
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning per-behavior embeddings and combined embeddings.

        Returns:
            behavior_embs: dict mapping behavior name to full embedding tensor
                           shape (num_users + num_items, embedding_dim)
            (user_emb, item_emb): combined weighted-sum embeddings split by role
        """
        weight_sum = sum(BEHAVIOR_WEIGHTS.values())
        behavior_embs: dict[str, torch.Tensor] = {}
        current_init = self.embedding.weight

        for b_idx, behavior in enumerate(BEHAVIORS):
            adj = adj_matrices[behavior]
            emb = self._propagate(adj, current_init)
            behavior_embs[behavior] = emb
            # Cascade: transform to next behavior's initial embeddings
            if b_idx < len(BEHAVIORS) - 1:
                current_init = self.transforms[b_idx](emb.detach())

        # Weighted sum for final embeddings
        final_emb = sum(BEHAVIOR_WEIGHTS[b] * emb for b, emb in behavior_embs.items())
        final_emb = final_emb / weight_sum
        user_emb = final_emb[: self.num_users]
        item_emb = final_emb[self.num_users :]
        return behavior_embs, (user_emb, item_emb)


def bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    base_embeddings: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    """BPR loss with L2 regularization on base embeddings."""
    pos_score = (user_emb[users] * item_emb[pos_items]).sum(dim=-1)
    neg_score = (user_emb[users] * item_emb[neg_items]).sum(dim=-1)
    bpr = -F.logsigmoid(pos_score - neg_score).mean()
    reg = base_embeddings.norm(p=2).pow(2) * lambda_reg / 2
    return bpr + reg


def compute_metrics(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_pairs: list[tuple[int, int]],
    train_pairs_set: set[tuple[int, int]],
    k_values: list[int],
    num_items: int,
) -> dict[str, float]:
    """Compute Precision@k for each k value."""
    if not test_pairs:
        return {f"precision@{k}": 0.0 for k in k_values}

    # Group test pairs by user
    user_pos: dict[int, set[int]] = {}
    for u, i in test_pairs:
        user_pos.setdefault(u, set()).add(i)

    max_k = max(k_values)
    precisions: dict[int, list[float]] = {k: [] for k in k_values}

    with torch.no_grad():
        for user_idx, pos_items in user_pos.items():
            scores = user_emb[user_idx] @ item_emb.T  # [num_items]
            # Mask out train items
            for item_idx in range(num_items):
                if (user_idx, item_idx) in train_pairs_set:
                    scores[item_idx] = -float("inf")
            _, topk_indices = torch.topk(scores, min(max_k, num_items))
            topk_list = topk_indices.cpu().tolist()
            for k in k_values:
                hits = sum(1 for idx in topk_list[:k] if idx in pos_items)
                precisions[k].append(hits / k)

    return {
        f"precision@{k}": float(np.mean(precisions[k])) if precisions[k] else 0.0 for k in k_values
    }


def _build_behavior_pairs(
    df: pd.DataFrame,
) -> dict[str, list[tuple[int, int]]]:
    """Extract (user_idx, client_idx) positive pairs per behavior."""
    pairs: dict[str, list[tuple[int, int]]] = {}
    for behavior, date_col in BEHAVIOR_DATE_MAP.items():
        filtered = df[df[date_col].notna()]
        pairs[behavior] = list(zip(filtered["user_idx"].values, filtered["client_idx"].values))
    return pairs


def _sample_negatives(
    pairs: list[tuple[int, int]],
    num_items: int,
    pos_set: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one negative item per positive pair."""
    neg_items = np.empty(len(pairs), dtype=np.int64)
    for i, (u, _) in enumerate(pairs):
        while True:
            neg = rng.integers(0, num_items)
            if (u, neg) not in pos_set:
                neg_items[i] = neg
                break
    return neg_items


def train(
    model: nn.Module,
    adj_matrices: dict[str, torch.sparse.FloatTensor],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ModelConfig,
    eval_config: EvalConfig,
    model_save_path: str = "models/best_model.pt",
    experiment_name: str = "lightgcn",
) -> dict[str, list[float]]:
    """Train MultiBehaviorLightGCN with BPR loss, early stopping, and MLflow tracking."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(42)

    # Build per-behavior positive pairs from training data
    train_behavior_pairs = _build_behavior_pairs(train_df)
    train_behavior_sets: dict[str, set[tuple[int, int]]] = {
        b: set(pairs) for b, pairs in train_behavior_pairs.items()
    }

    # Validation pairs (apply behavior for early stopping)
    val_apply_pairs = list(
        zip(
            val_df[val_df["apply_date"].notna()]["user_idx"].values,
            val_df[val_df["apply_date"].notna()]["client_idx"].values,
        )
    )
    train_apply_set = train_behavior_sets.get("apply", set())

    num_items = model.num_items
    history: dict[str, list[float]] = {"train_loss": [], "val_precision@10": []}
    best_precision = -1.0
    patience_counter = 0

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "embedding_dim": config.embedding_dim,
                "num_layers": config.num_layers,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "patience": config.patience,
                "lambda_reg": config.lambda_reg,
                "model_type": type(model).__name__,
            }
        )

        weight_sum = sum(BEHAVIOR_WEIGHTS.values())

        for epoch in range(config.epochs):
            model.train()
            epoch_losses: list[float] = []

            # Determine max steps across all behaviors for one epoch
            max_steps = (
                max(
                    (len(pairs) + config.batch_size - 1) // config.batch_size
                    for pairs in train_behavior_pairs.values()
                    if pairs
                )
                if any(train_behavior_pairs.values())
                else 0
            )

            for _ in range(max_steps):
                # One forward pass for all behaviors (efficient)
                if isinstance(model, MultiBehaviorLightGCN):
                    behavior_embs, (user_emb, item_emb) = model(adj_matrices)
                else:
                    user_emb, item_emb = model(adj_matrices.get("apply"))
                    behavior_embs = {"apply": torch.cat([user_emb, item_emb], dim=0)}

                total_loss = torch.tensor(0.0, device=device)

                for behavior, pairs in train_behavior_pairs.items():
                    if not pairs:
                        continue
                    w = BEHAVIOR_WEIGHTS[behavior]
                    pos_set = train_behavior_sets[behavior]
                    batch_size = min(config.batch_size, len(pairs))
                    batch_idx = rng.choice(len(pairs), size=batch_size, replace=False)
                    batch_pairs = [pairs[i] for i in batch_idx]
                    neg = _sample_negatives(batch_pairs, num_items, pos_set, rng)

                    users_t = torch.tensor(
                        [p[0] for p in batch_pairs], dtype=torch.long, device=device
                    )
                    pos_t = torch.tensor(
                        [p[1] for p in batch_pairs], dtype=torch.long, device=device
                    )
                    neg_t = torch.tensor(neg, dtype=torch.long, device=device)

                    # Use per-behavior embeddings for multi-task learning
                    b_emb = behavior_embs[behavior]
                    b_user_emb = b_emb[: model.num_users]
                    b_item_emb = b_emb[model.num_users :]

                    loss = bpr_loss(
                        b_user_emb,
                        b_item_emb,
                        users_t,
                        pos_t,
                        neg_t,
                        model.embedding.weight,
                        config.lambda_reg,
                    )
                    total_loss = total_loss + loss * w / weight_sum

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(avg_loss)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    if isinstance(model, MultiBehaviorLightGCN):
                        _, (user_emb, item_emb) = model(adj_matrices)
                    else:
                        user_emb, item_emb = model(adj_matrices.get("apply"))

                    metrics = compute_metrics(
                        user_emb,
                        item_emb,
                        val_apply_pairs,
                        train_apply_set,
                        eval_config.k_values,
                        num_items,
                    )
                val_p10 = metrics.get("precision@10", 0.0)
                history["val_precision@10"].append(val_p10)
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=epoch)

                if val_p10 > best_precision:
                    best_precision = val_p10
                    patience_counter = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience_counter += 10  # evaluate every 10 epochs

                if patience_counter >= config.patience:
                    break

        # Log best model artifact
        if os.path.exists(model_save_path):
            mlflow.log_artifact(model_save_path, artifact_path="model")
        mlflow.log_metric("best_val_precision@10", best_precision)

    return history
