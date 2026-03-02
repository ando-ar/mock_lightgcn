from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train_ratio: float = 0.8
    num_users: int = 500
    num_clients: int = 200
    num_recommendations_per_user_lambda: float = 20.0
    p_apply: float = 0.3
    p_pass: float = 0.5
    p_offer: float = 0.4
    p_offer_accept: float = 0.6
    seed: int = 42


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 200
    patience: int = 20
    lambda_reg: float = 1e-4


@dataclass
class EvalConfig:
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class BehaviorWeights:
    offer_accept: float = 4.0
    offer: float = 3.0
    p_pass: float = 2.0
    apply: float = 1.0
