import torch

from mock_lightgcn.config import DataConfig, ModelConfig
from mock_lightgcn.dataset import encode_ids, generate_dummy_data, split_train_test
from mock_lightgcn.features import build_adj_matrix
from mock_lightgcn.modeling.train import (
    LightGCN,
    MultiBehaviorLightGCN,
    bpr_loss,
)

BEHAVIORS = ["apply", "pass", "offer", "offer_accept"]


def _make_data(seed: int = 0):
    cfg = DataConfig(num_users=30, num_clients=20, seed=seed)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df)
    train_enc, test_enc, _, _ = encode_ids(train_df, test_df)
    num_users = int(train_enc["user_idx"].nunique())
    num_clients = int(train_enc["client_idx"].nunique())
    adj_matrices = {b: build_adj_matrix(train_enc, num_users, num_clients, b) for b in BEHAVIORS}
    return train_enc, test_enc, num_users, num_clients, adj_matrices


def test_lightgcn_forward_shape():
    train_df, _, num_users, num_clients, adj_matrices = _make_data()
    cfg = ModelConfig(embedding_dim=16, num_layers=2)
    model = LightGCN(num_users, num_clients, cfg)
    adj = adj_matrices["apply"]
    user_emb, item_emb = model(adj)
    assert user_emb.shape == (num_users, 16)
    assert item_emb.shape == (num_clients, 16)


def test_lightgcn_forward_no_nan():
    train_df, _, num_users, num_clients, adj_matrices = _make_data()
    cfg = ModelConfig(embedding_dim=16, num_layers=2)
    model = LightGCN(num_users, num_clients, cfg)
    adj = adj_matrices["apply"]
    user_emb, item_emb = model(adj)
    assert not torch.isnan(user_emb).any()
    assert not torch.isnan(item_emb).any()


def test_multi_behavior_lightgcn_forward_shape():
    train_df, _, num_users, num_clients, adj_matrices = _make_data()
    cfg = ModelConfig(embedding_dim=16, num_layers=2)
    model = MultiBehaviorLightGCN(num_users, num_clients, cfg)
    behavior_embs, (user_emb, item_emb) = model(adj_matrices)
    assert user_emb.shape == (num_users, 16)
    assert item_emb.shape == (num_clients, 16)
    assert set(behavior_embs.keys()) == {"apply", "pass", "offer", "offer_accept"}


def test_multi_behavior_lightgcn_forward_no_nan():
    train_df, _, num_users, num_clients, adj_matrices = _make_data()
    cfg = ModelConfig(embedding_dim=16, num_layers=2)
    model = MultiBehaviorLightGCN(num_users, num_clients, cfg)
    behavior_embs, (user_emb, item_emb) = model(adj_matrices)
    assert not torch.isnan(user_emb).any()
    assert not torch.isnan(item_emb).any()
    for emb in behavior_embs.values():
        assert not torch.isnan(emb).any()


def test_multi_behavior_lightgcn_embeddings_differ_from_lightgcn():
    """MB-LightGCN should produce different embeddings than single-behavior LightGCN."""
    train_df, _, num_users, num_clients, adj_matrices = _make_data(seed=5)
    cfg = ModelConfig(embedding_dim=16, num_layers=1)
    mb_model = MultiBehaviorLightGCN(num_users, num_clients, cfg)
    base_model = LightGCN(num_users, num_clients, cfg)

    # Copy shared embeddings to make comparison meaningful
    base_model.embedding.weight.data.copy_(mb_model.embedding.weight.data)

    _, (mb_user, _) = mb_model(adj_matrices)
    base_user, _ = base_model(adj_matrices["apply"])

    # They should differ because MB uses multiple behaviors and cascade
    if any(adj_matrices[b]._nnz() > 0 for b in BEHAVIORS if b != "apply"):
        assert not torch.allclose(mb_user, base_user, atol=1e-6)


def test_bpr_loss_positive():
    num_users, num_items, dim = 10, 15, 8
    user_emb = torch.randn(num_users, dim)
    item_emb = torch.randn(num_items, dim)
    base_emb = torch.randn(num_users + num_items, dim)
    users = torch.tensor([0, 1, 2])
    pos_items = torch.tensor([3, 5, 7])
    neg_items = torch.tensor([1, 2, 4])

    loss = bpr_loss(user_emb, item_emb, users, pos_items, neg_items, base_emb, lambda_reg=1e-4)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_bpr_loss_decreases_with_better_scores():
    """Loss should be lower when positive scores > negative scores."""
    dim = 8
    user_emb = torch.zeros(5, dim)
    item_emb = torch.zeros(10, dim)
    base_emb = torch.zeros(15, dim)

    # Make pos score high, neg score low
    item_emb_good = item_emb.clone()
    item_emb_good[3] = torch.ones(dim)  # pos
    item_emb_good[1] = -torch.ones(dim)  # neg
    user_emb_good = user_emb.clone()
    user_emb_good[0] = torch.ones(dim)

    users = torch.tensor([0])
    pos_items = torch.tensor([3])
    neg_items = torch.tensor([1])

    loss_good = bpr_loss(
        user_emb_good, item_emb_good, users, pos_items, neg_items, base_emb, lambda_reg=0.0
    )
    loss_bad = bpr_loss(
        user_emb_good, -item_emb_good, users, pos_items, neg_items, base_emb, lambda_reg=0.0
    )
    assert loss_good.item() < loss_bad.item()


def test_bpr_loss_backward():
    """Verify gradients flow through bpr_loss."""
    num_users, num_items, dim = 5, 8, 4
    user_emb = torch.randn(num_users, dim, requires_grad=True)
    item_emb = torch.randn(num_items, dim, requires_grad=True)
    base_emb = torch.randn(num_users + num_items, dim, requires_grad=True)
    users = torch.tensor([0, 1])
    pos_items = torch.tensor([2, 3])
    neg_items = torch.tensor([4, 5])

    loss = bpr_loss(user_emb, item_emb, users, pos_items, neg_items, base_emb, lambda_reg=1e-4)
    loss.backward()
    assert user_emb.grad is not None
    assert item_emb.grad is not None
