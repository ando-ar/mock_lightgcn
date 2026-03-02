import pandas as pd

from mock_lightgcn.config import DataConfig
from mock_lightgcn.dataset import encode_ids, generate_dummy_data, split_train_test


def test_generate_dummy_data_shape():
    cfg = DataConfig(num_users=10, num_clients=20, seed=0)
    df = generate_dummy_data(cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {
        "user_id",
        "client_id",
        "recommend_date",
        "apply_date",
        "pass_date",
        "offer_date",
        "offer_accept_date",
    }
    assert expected_cols.issubset(df.columns)


def test_generate_dummy_data_no_duplicate_pairs():
    cfg = DataConfig(num_users=50, num_clients=30, seed=1)
    df = generate_dummy_data(cfg)
    assert df.duplicated(subset=["user_id", "client_id"]).sum() == 0


def test_generate_dummy_data_at_least_one_rec_per_user():
    cfg = DataConfig(num_users=10, num_clients=20, seed=2)
    df = generate_dummy_data(cfg)
    assert df["user_id"].nunique() == cfg.num_users


def test_generate_dummy_data_offer_accept_at_most_one_per_user():
    cfg = DataConfig(num_users=50, num_clients=30, seed=3)
    df = generate_dummy_data(cfg)
    # Each user can have at most 1 offer_accept
    counts = df.groupby("user_id")["offer_accept_date"].apply(lambda x: x.notna().sum())
    assert (counts <= 1).all()


def test_generate_dummy_data_behavior_dependencies():
    cfg = DataConfig(num_users=100, num_clients=50, seed=4)
    df = generate_dummy_data(cfg)
    # pass requires apply
    assert (df["pass_date"].notna() & df["apply_date"].isna()).sum() == 0
    # offer requires pass
    assert (df["offer_date"].notna() & df["pass_date"].isna()).sum() == 0
    # offer_accept requires offer
    assert (df["offer_accept_date"].notna() & df["offer_date"].isna()).sum() == 0


def test_split_train_test_ratio():
    cfg = DataConfig(num_users=50, num_clients=30, seed=0)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df, train_ratio=0.8)
    assert len(train_df) + len(test_df) == len(df)
    assert abs(len(train_df) / len(df) - 0.8) < 0.01


def test_split_train_test_sorted_by_date():
    cfg = DataConfig(num_users=50, num_clients=30, seed=0)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df, train_ratio=0.8)
    # All train dates should be <= all test dates
    assert train_df["recommend_date"].max() <= test_df["recommend_date"].min()


def test_encode_ids_zero_indexed():
    cfg = DataConfig(num_users=20, num_clients=10, seed=0)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df, train_ratio=0.8)
    train_enc, test_enc, user2idx, client2idx = encode_ids(train_df, test_df)

    assert train_enc["user_idx"].min() == 0
    assert train_enc["client_idx"].min() == 0
    assert train_enc["user_idx"].max() == len(user2idx) - 1
    assert train_enc["client_idx"].max() == len(client2idx) - 1


def test_encode_ids_cold_start_excluded():
    cfg = DataConfig(num_users=20, num_clients=10, seed=0)
    df = generate_dummy_data(cfg)
    train_df, test_df = split_train_test(df, train_ratio=0.8)
    train_enc, test_enc, user2idx, client2idx = encode_ids(train_df, test_df)

    # No unknown users or clients in test
    assert test_enc["user_idx"].notna().all()
    assert test_enc["client_idx"].notna().all()
    assert test_enc["user_id"].isin(user2idx).all()
    assert test_enc["client_id"].isin(client2idx).all()
