import numpy as np
import pandas as pd

from mock_lightgcn.config import DataConfig


def generate_dummy_data(config: DataConfig) -> pd.DataFrame:
    """Generate a realistic dummy recommendation dataset."""
    rng = np.random.default_rng(config.seed)

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    date_range_days = (end_date - start_date).days

    rows: list[dict] = []

    for user_id in range(config.num_users):
        n_recs = max(1, rng.poisson(config.num_recommendations_per_user_lambda))
        # Sample unique client_ids for this user
        n_recs = min(n_recs, config.num_clients)
        client_ids = rng.choice(config.num_clients, size=n_recs, replace=False)

        has_accepted = False

        for client_id in client_ids:
            recommend_date = start_date + pd.Timedelta(
                days=int(rng.integers(0, date_range_days + 1))
            )

            apply_date = pd.NaT
            pass_date = pd.NaT
            offer_date = pd.NaT
            offer_accept_date = pd.NaT

            if rng.random() < config.p_apply:
                apply_date = recommend_date + pd.Timedelta(days=int(rng.integers(1, 15)))

                if rng.random() < config.p_pass:
                    pass_date = apply_date + pd.Timedelta(days=int(rng.integers(1, 30)))

                    if rng.random() < config.p_offer:
                        offer_date = pass_date + pd.Timedelta(days=int(rng.integers(1, 14)))

                        if not has_accepted and rng.random() < config.p_offer_accept:
                            offer_accept_date = offer_date + pd.Timedelta(
                                days=int(rng.integers(1, 7))
                            )
                            has_accepted = True

            rows.append(
                {
                    "user_id": user_id,
                    "client_id": int(client_id),
                    "recommend_date": recommend_date,
                    "apply_date": apply_date,
                    "pass_date": pass_date,
                    "offer_date": offer_date,
                    "offer_accept_date": offer_accept_date,
                }
            )

    df = pd.DataFrame(rows)
    return df


def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by recommend_date: first train_ratio fraction is train.

    Cold-start users (in test but not in train) are excluded from test.
    """
    df_sorted = df.sort_values("recommend_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Exclude cold-start users from test
    train_user_ids = set(train_df["user_id"].unique())
    test_df = test_df[test_df["user_id"].isin(train_user_ids)].copy()

    return train_df, test_df


def encode_ids(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Create 0-indexed mappings from train_df and filter test_df to known IDs."""
    # Build mappings from train users/clients
    train_users = sorted(train_df["user_id"].unique())
    train_clients = sorted(train_df["client_id"].unique())
    user2idx = {uid: idx for idx, uid in enumerate(train_users)}
    client2idx = {cid: idx for idx, cid in enumerate(train_clients)}

    # Filter test_df to only rows with known user_id and client_id
    test_df = test_df[
        test_df["user_id"].isin(user2idx) & test_df["client_id"].isin(client2idx)
    ].copy()

    # Add index columns
    train_df = train_df.copy()
    train_df["user_idx"] = train_df["user_id"].map(user2idx)
    train_df["client_idx"] = train_df["client_id"].map(client2idx)
    test_df["user_idx"] = test_df["user_id"].map(user2idx)
    test_df["client_idx"] = test_df["client_id"].map(client2idx)

    return train_df, test_df, user2idx, client2idx
