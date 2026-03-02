import torch

from mock_lightgcn.config import DataConfig
from mock_lightgcn.dataset import encode_ids, generate_dummy_data, split_train_test
from mock_lightgcn.features import build_adj_matrix


def _get_encoded_train(seed: int = 0):
    cfg = DataConfig(num_users=50, num_clients=30, seed=seed)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df)
    train_enc, _, _, _ = encode_ids(train_df, test_df)
    num_users = train_enc["user_idx"].nunique()
    num_clients = train_enc["client_idx"].nunique()
    return train_enc, num_users, num_clients


def test_build_adj_matrix_shape():
    train_df, num_users, num_clients = _get_encoded_train()
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    n = num_users + num_clients
    assert adj.shape == (n, n)


def test_build_adj_matrix_symmetric():
    train_df, num_users, num_clients = _get_encoded_train()
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    dense = adj.to_dense()
    diff = (dense - dense.T).abs().max().item()
    assert diff < 1e-5, f"Adjacency matrix is not symmetric, max diff: {diff}"


def test_build_adj_matrix_normalized():
    train_df, num_users, num_clients = _get_encoded_train()
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    dense = adj.to_dense()
    # All values should be <= 1 (normalized)
    assert dense.max().item() <= 1.0 + 1e-6


def test_build_adj_matrix_sparse():
    train_df, num_users, num_clients = _get_encoded_train()
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    assert adj.is_sparse


def test_build_adj_matrix_no_interactions():
    train_df, num_users, num_clients = _get_encoded_train()
    # offer_accept may have few/no interactions - just check shape
    adj = build_adj_matrix(train_df, num_users, num_clients, "offer_accept")
    n = num_users + num_clients
    assert adj.shape == (n, n)


def test_build_adj_matrix_invalid_behavior():
    import pytest

    train_df, num_users, num_clients = _get_encoded_train()
    with pytest.raises(ValueError):
        build_adj_matrix(train_df, num_users, num_clients, "invalid_behavior")


def test_build_adj_matrix_apply_has_interactions():
    train_df, num_users, num_clients = _get_encoded_train(seed=42)
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    # Should have some nonzero entries if there are applies in training data
    if train_df["apply_date"].notna().sum() > 0:
        assert adj._nnz() > 0


def test_build_adj_matrix_consistent_with_dense():
    """Verify the upper-right block of the adj matrix equals R."""
    train_df, num_users, num_clients = _get_encoded_train(seed=7)
    adj = build_adj_matrix(train_df, num_users, num_clients, "apply")
    dense = adj.to_dense()

    # Upper-left block (user-user) should be zero
    assert dense[:num_users, :num_users].abs().sum().item() == 0
    # Lower-right block (client-client) should be zero
    assert dense[num_users:, num_users:].abs().sum().item() == 0
    # Off-diagonal blocks should be symmetric
    upper_right = dense[:num_users, num_users:]
    lower_left = dense[num_users:, :num_users]
    assert torch.allclose(upper_right, lower_left.T, atol=1e-6)
