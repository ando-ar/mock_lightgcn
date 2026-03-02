import pandas as pd
import torch

BEHAVIOR_DATE_MAP = {
    "apply": "apply_date",
    "pass": "pass_date",
    "offer": "offer_date",
    "offer_accept": "offer_accept_date",
}


def build_adj_matrix(
    df: pd.DataFrame,
    num_users: int,
    num_clients: int,
    behavior: str,
) -> torch.sparse.FloatTensor:
    """Build a symmetrically normalized bipartite adjacency matrix.

    The matrix has shape (num_users + num_clients, num_users + num_clients):
        [  0    R  ]
        [ R^T   0  ]
    where R[u, c] = 1 if the user-client pair has the given behavior.
    Returns D^{-1/2} A D^{-1/2} as a sparse FloatTensor.
    """
    if behavior not in BEHAVIOR_DATE_MAP:
        raise ValueError(f"behavior must be one of {list(BEHAVIOR_DATE_MAP)}, got '{behavior}'")

    date_col = BEHAVIOR_DATE_MAP[behavior]
    filtered = df[df[date_col].notna()]

    user_indices = filtered["user_idx"].values
    client_indices = filtered["client_idx"].values

    n = num_users + num_clients

    # Build symmetric adjacency: upper-right block R and lower-left block R^T
    row = torch.cat(
        [
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(client_indices + num_users, dtype=torch.long),
        ]
    )
    col = torch.cat(
        [
            torch.tensor(client_indices + num_users, dtype=torch.long),
            torch.tensor(user_indices, dtype=torch.long),
        ]
    )
    values = torch.ones(len(row), dtype=torch.float32)

    adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (n, n)).coalesce()

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.zeros(n)
    nonzero_mask = degree > 0
    d_inv_sqrt[nonzero_mask] = degree[nonzero_mask].pow(-0.5)

    # Scale each entry: A_ij * d_inv_sqrt[i] * d_inv_sqrt[j]
    indices = adj.indices()
    vals = adj.values()
    vals = vals * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]

    norm_adj = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()
    return norm_adj
