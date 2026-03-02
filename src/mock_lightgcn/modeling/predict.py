from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from mock_lightgcn.modeling.train import MultiBehaviorLightGCN


def get_recommendations(
    model: nn.Module,
    adj_matrices: dict[str, torch.sparse.FloatTensor],
    user_ids: list[int],
    train_df: pd.DataFrame,
    num_clients: int,
    k: int = 10,
) -> dict[int, list[int]]:
    """Get top-k recommended client indices for each user.

    Excludes clients already interacted with in train_df.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, MultiBehaviorLightGCN):
            _, (user_emb, item_emb) = model(adj_matrices)
        else:
            user_emb, item_emb = model(adj_matrices.get("apply"))

    # Build per-user exclusion set from train_df
    train_interactions: dict[int, set[int]] = {}
    for _, row in train_df[["user_idx", "client_idx"]].iterrows():
        u = int(row["user_idx"])
        c = int(row["client_idx"])
        train_interactions.setdefault(u, set()).add(c)

    recommendations: dict[int, list[int]] = {}
    with torch.no_grad():
        for user_idx in user_ids:
            scores = user_emb[user_idx] @ item_emb.T
            exclude = train_interactions.get(user_idx, set())
            for c in exclude:
                scores[c] = -float("inf")
            _, topk = torch.topk(scores, min(k, num_clients))
            recommendations[user_idx] = topk.cpu().tolist()

    return recommendations
