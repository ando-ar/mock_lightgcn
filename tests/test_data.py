# Smoke test: verify the package imports correctly
from mock_lightgcn.config import DataConfig, EvalConfig, ModelConfig


def test_config_defaults():
    data_cfg = DataConfig()
    assert data_cfg.num_users == 500
    assert data_cfg.num_clients == 200
    assert data_cfg.train_ratio == 0.8

    model_cfg = ModelConfig()
    assert model_cfg.embedding_dim == 64
    assert model_cfg.num_layers == 3

    eval_cfg = EvalConfig()
    assert eval_cfg.k_values == [5, 10, 20]
