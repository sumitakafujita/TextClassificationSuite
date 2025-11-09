"""バックエンド全体で共有する設定値とパラメータ定義."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EmbeddingConfig:
    """埋め込みモデルに関する設定."""

    model_name: str = "intfloat/multilingual-e5-small"
    instruction: str = "query: "
    batch_size: int = 32


@dataclass(frozen=True)
class ModelingConfig:
    """モデリングおよび評価の基本設定."""

    cv_splits: int = 5
    random_seed: int = 42
    logistic_trials: int = 30
    bagging_trials: int = 30
    lightgbm_trials: int = 40
    mlp_trials: int = 30
    linear_svm_trials: int = 35
    low_confidence_threshold: float = 0.85
    similarity_threshold: float = 0.92
    similarity_top_k: int = 3


@dataclass(frozen=True)
class DataConfig:
    """データ入出力に関するパス設定."""

    data_path: Path = Path("data/text_classification_samples_200.csv")
    text_column: str = "text"
    label_column: str = "category_label"


@dataclass(frozen=True)
class BackendConfig:
    """バックエンド全体のルート設定."""

    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    candidate_models: List[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "bagging_log_regression",
            "mlp_classifier",
            "linear_svm",
            "lightgbm_classifier",
        ]
    )
