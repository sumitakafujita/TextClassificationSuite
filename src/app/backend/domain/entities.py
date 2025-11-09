"""ドメイン層で利用するデータクラス定義."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetArtifacts:
    """データセットにまつわる情報一式."""

    lazy_frame: pl.LazyFrame
    records: pd.DataFrame
    labels: np.ndarray
    label_encoder: LabelEncoder
    text_column: str
    label_column: str


@dataclass
class EmbeddingArtifacts:
    """埋め込みベクトルと関連情報."""

    text_embeddings: np.ndarray
    augmented_embeddings: np.ndarray
    category_embeddings: np.ndarray


@dataclass
class ModelSelectionResult:
    """モデル選定過程の結果."""

    best_model_name: str
    best_estimator: BaseEstimator
    cv_summary: pd.DataFrame
    study_summaries: Dict[str, Dict[str, float]]


@dataclass
class ConfidenceReport:
    """確信度評価と低確信度サンプル."""

    detail: pd.DataFrame
    low_confidence: pd.DataFrame


@dataclass
class SimilarityGroup:
    """類似問い合わせグループ."""

    root_id: int
    root_text: str
    predicted_label: str
    neighbors: List[Dict[str, object]]
